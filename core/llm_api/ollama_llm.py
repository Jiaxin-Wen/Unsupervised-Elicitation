import asyncio
import json
import logging
import time
from datetime import datetime
from traceback import format_exc
from typing import Optional, Union, List

import attrs
from termcolor import cprint

from core.llm_api.base_llm import PRINT_COLORS, LLMResponse, ModelAPIProtocol

try:
    import ollama
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None
    OllamaLLM = None
    PromptTemplate = None
    HumanMessage = SystemMessage = AIMessage = None
    print("Warning: Ollama and LangChain not available. Install with: pip install ollama langchain_ollama langchain_core")

OllamaChatPrompt = list[dict[str, str]]
LOGGER = logging.getLogger(__name__)

# Common Ollama models - these are local models that can be pulled via `ollama pull <model>`
OLLAMA_MODELS = {
    # Code models
    "llama3.2:latest",
    "llama3.2:3b",
    "llama3.1:latest", 
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "llama3:latest",
    "llama3:8b",
    "llama3:70b",
    "codellama:latest",
    "codellama:7b",
    "codellama:13b",
    "codellama:34b",
    "codeqwen:latest",
    "codeqwen:7b",
    "deepseek-coder:latest",
    "deepseek-coder:6.7b",
    "deepseek-coder:33b",
    
    # General purpose models
    "mistral:latest",
    "mistral:7b",
    "mixtral:latest",
    "mixtral:8x7b",
    "mixtral:8x22b",
    "qwen2.5:latest",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "qwen2.5:72b",
    "gemma2:latest",
    "gemma2:2b",
    "gemma2:9b",
    "gemma2:27b",
    "phi3:latest",
    "phi3:3.8b",
    "phi3:14b",
    "neural-chat:latest",
    "neural-chat:7b",
    "starling-lm:latest",
    "starling-lm:7b",
    
    # Specialized models
    "orca-mini:latest",
    "orca-mini:3b",
    "orca-mini:7b",
    "orca-mini:13b",
    "vicuna:latest",
    "vicuna:7b",
    "vicuna:13b",
    "vicuna:33b",
    "wizardcoder:latest",
    "wizardcoder:13b",
    "wizardcoder:34b",
    "starcoder:latest",
    "starcoder:7b",
    "starcoder:15b",
}


def price_per_token(model_id: str) -> tuple[float, float]:
    """
    Returns the (input token, output token) price for Ollama models.
    Since Ollama runs locally, costs are essentially zero.
    """
    return (0.0, 0.0)


def count_tokens_estimate(text: str) -> int:
    """
    Rough estimate of token count for Ollama models.
    This is a simple approximation - Ollama models may have different tokenizers.
    """
    # Simple approximation: ~4 characters per token
    return len(text) // 4


def format_messages_for_ollama(messages: OllamaChatPrompt) -> str:
    """
    Convert OpenAI-style messages to a single prompt string for Ollama.
    """
    formatted_parts = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"Human: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(formatted_parts) + "\n\nAssistant:"


def extract_system_prompt(messages: OllamaChatPrompt) -> tuple[str, list[dict]]:
    """
    Extract system prompt and return remaining messages.
    """
    system_prompt = ""
    remaining_messages = []
    
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
        else:
            remaining_messages.append(message)
    
    return system_prompt, remaining_messages


@attrs.define()
class OllamaModel(ModelAPIProtocol):
    host: str = "http://localhost:11434"  # Default Ollama host
    temperature: float = 0.0
    print_prompt_and_response: bool = False
    
    def __attrs_post_init__(self):
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama and LangChain are required. Install with: pip install ollama langchain_ollama langchain_core")
        
        # Test connection to Ollama
        try:
            if ollama is not None:
                ollama.list()
        except Exception as e:
            LOGGER.warning(f"Could not connect to Ollama at {self.host}: {e}")
            LOGGER.warning("Make sure Ollama is running: 'ollama serve'")

    def _assert_valid_id(self, model_id: str):
        """Check if model is available in Ollama."""
        if ollama is None:
            LOGGER.warning("Ollama not available, cannot check model availability")
            return
            
        try:
            models_response = ollama.list()
            available_models = []
            
            # Handle ollama._types.ListResponse
            if hasattr(models_response, 'models'):
                # ollama._types.ListResponse has a models attribute
                for model in models_response.models:
                    if hasattr(model, 'name'):
                        available_models.append(model.name)
                    elif hasattr(model, 'model'):
                        available_models.append(model.model)
                    elif isinstance(model, dict):
                        name = model.get('name') or model.get('model') or model.get('id') or str(model)
                        available_models.append(name)
                    else:
                        available_models.append(str(model))
            elif isinstance(models_response, dict):
                if 'models' in models_response:
                    # Format: {'models': [{'name': 'model_name'}, ...]}
                    for model in models_response['models']:
                        if isinstance(model, dict):
                            name = model.get('name') or model.get('model') or model.get('id') or str(model)
                            available_models.append(name)
                        else:
                            available_models.append(str(model))
                else:
                    # Format: {'model1': ..., 'model2': ...} or similar
                    available_models = list(models_response.keys())
            elif isinstance(models_response, list):
                # Format: ['model1', 'model2', ...] or [{'name': 'model1'}, ...]
                for model in models_response:
                    if isinstance(model, dict):
                        name = model.get('name') or model.get('model') or model.get('id') or str(model)
                        available_models.append(name)
                    else:
                        available_models.append(str(model))
            else:
                # Try to get models attribute or convert to string
                if hasattr(models_response, '__dict__'):
                    LOGGER.debug(f"Unknown response type with attributes: {list(models_response.__dict__.keys())}")
                LOGGER.warning(f"Unexpected models response type: {type(models_response)}")
                return
            
            # Handle cases where model_id might have :latest suffix or not
            model_variants = [model_id, f"{model_id}:latest"]
            if ":" in model_id:
                base_name = model_id.split(":")[0]
                model_variants.append(base_name)
                model_variants.append(f"{base_name}:latest")
            
            model_found = False
            for variant in model_variants:
                if any(variant in available_model or available_model in variant for available_model in available_models):
                    model_found = True
                    break
            
            if not model_found:
                LOGGER.warning(f"Model {model_id} not found locally. Available models: {available_models}")
                LOGGER.warning(f"You can pull it with: ollama pull {model_id}")
                
        except Exception as e:
            LOGGER.warning(f"Could not check available models: {e}")
            # Log the actual response for debugging
            try:
                models_response = ollama.list()
                LOGGER.debug(f"Ollama models response type: {type(models_response)}")
                if hasattr(models_response, '__dict__'):
                    LOGGER.debug(f"Response attributes: {list(models_response.__dict__.keys())}")
            except:
                pass

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_ollama_prompt.txt"
        with open(f"prompt_history/{filename}", "w") as f:
            if isinstance(prompt, list):
                json_str = json.dumps(prompt, indent=4)
            else:
                json_str = prompt
            f.write(json_str)
        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, response):
        with open(f"prompt_history/{prompt_file}", "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(response.to_dict(), indent=4)
            f.write(json_str)

    async def _make_api_call(
        self, prompt: Union[str, OllamaChatPrompt], model_id: str, start_time: float, **kwargs
    ) -> list[LLMResponse]:
        """Make API call to Ollama."""
        if ollama is None:
            raise RuntimeError("Ollama not available. Install with: pip install ollama")
            
        LOGGER.debug(f"Making {model_id} call to Ollama")
        
        # Process prompt
        if isinstance(prompt, list):
            # Convert chat messages to single prompt
            formatted_prompt = format_messages_for_ollama(prompt)
        else:
            formatted_prompt = prompt
        
        # Extract parameters
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            api_start = time.time()
            
            # Use Ollama directly for synchronous call, then wrap in async
            def sync_ollama_call():
                response = ollama.generate(
                    model=model_id,
                    prompt=formatted_prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
                return response
            
            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, sync_ollama_call)
            
            api_duration = time.time() - api_start
            duration = time.time() - start_time
            
            # Extract response text
            completion = response.get('response', '')
            
            # Determine stop reason
            if response.get('done_reason') == 'length':
                stop_reason = 'max_tokens'
            else:
                stop_reason = 'stop'
            
            # Create LLM response
            llm_response = LLMResponse(
                model_id=model_id,
                completion=completion,
                stop_reason=stop_reason,
                duration=duration,
                api_duration=api_duration,
                cost=0.0,  # Ollama is free/local
                logprobs=None,  # Ollama doesn't provide logprobs by default
            )
            
            return [llm_response]
            
        except Exception as e:
            error_msg = f"Ollama API call failed: {str(e)}"
            LOGGER.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def _print_prompt_and_response(prompt: Union[str, OllamaChatPrompt], responses: list[LLMResponse]):
        """Print prompt and response for debugging."""
        if isinstance(prompt, list):
            # Chat format
            for message in prompt:
                role = message["role"]
                content = message["content"]
                cprint(f"=={role.upper()}:", "white")
                cprint(content, PRINT_COLORS.get(role, "white"))
        else:
            # String format
            cprint("==PROMPT:", "white")
            cprint(prompt, PRINT_COLORS["user"])
        
        for i, response in enumerate(responses):
            if len(responses) > 1:
                cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            else:
                cprint(f"==RESPONSE ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()

    async def __call__(
        self,
        model_ids: list[str],
        prompt: Union[str, OllamaChatPrompt],
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make API call to Ollama model.
        
        Args:
            model_ids: List of model IDs (Ollama only uses the first one)
            prompt: The prompt to send (string or chat messages)
            print_prompt_and_response: Whether to print debug info
            max_attempts: Number of retry attempts
            **kwargs: Additional parameters
        """
        start = time.time()
        
        # Ollama doesn't support multiple models simultaneously, use first one
        if len(model_ids) > 1:
            LOGGER.warning(f"Ollama only supports one model at a time. Using {model_ids[0]}")
        model_id = model_ids[0]
        
        # Validate model
        self._assert_valid_id(model_id)
        
        # Create prompt history file
        # prompt_file = self._create_prompt_history_file(prompt)
        
        responses: Optional[list[LLMResponse]] = None
        
        for attempt in range(max_attempts):
            try:
                responses = await self._make_api_call(prompt, model_id, start, **kwargs)
                break
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warning(f"Ollama API error (attempt {attempt + 1}): {error_info}")
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1.5 ** attempt)
                else:
                    raise RuntimeError(f"Failed to get response from Ollama after {max_attempts} attempts") from e
        
        if responses is None:
            raise RuntimeError(f"Failed to get response from Ollama after {max_attempts} attempts")
        
        # Print debug info if requested
        if self.print_prompt_and_response or print_prompt_and_response:
            self._print_prompt_and_response(prompt, responses)
        
        # Add response to prompt file
        # for response in responses:
        #     self._add_response_to_prompt_file(prompt_file, response)
        
        duration = time.time() - start
        LOGGER.debug(f"Completed Ollama call to {model_id} in {duration:.2f}s")
        
        # Return in the expected format for the codebase
        return [{"prompt": prompt, "response": response.to_dict()} for response in responses]


# Alias for consistency with other model classes
OllamaChatModel = OllamaModel

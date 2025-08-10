import asyncio
import json
import logging
import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import attrs

from core.llm_api.anthropic_llm import ANTHROPIC_MODELS, AnthropicChatModel
from core.llm_api.base_llm import LLMResponse, ModelAPIProtocol
from core.llm_api.openai_llm import (
    BASE_MODELS,
    GPT_CHAT_MODELS,
    OAIBasePrompt,
    OAIChatPrompt,
    OpenAIBaseModel,
    OpenAIChatModel,
)
from core.llm_api.ollama_llm import OLLAMA_MODELS, OllamaModel
from core.utils import load_secrets

LOGGER = logging.getLogger(__name__)


@attrs.define()
class ModelAPI:
    anthropic_num_threads: int = 2  # current redwood limit is 5
    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    organization: str = "NYU_ORG"
    print_prompt_and_response: bool = False
    ollama_host: str = "http://localhost:11434"  # Default Ollama host
    ollama_temperature: float = 0.0

    _openai_base: OpenAIBaseModel = attrs.field(init=False)
    _openai_base_arg: OpenAIBaseModel = attrs.field(init=False)
    _openai_chat: OpenAIChatModel = attrs.field(init=False)
    _anthropic_chat: AnthropicChatModel = attrs.field(init=False)
    _ollama: OllamaModel = attrs.field(init=False)

    running_cost: float = attrs.field(init=False, default=0)
    model_timings: dict[str, list[float]] = attrs.field(init=False, default={})
    model_wait_times: dict[str, list[float]] = attrs.field(init=False, default={})

    def __attrs_post_init__(self):
        secrets = load_secrets()
        if self.organization is None:
            self.organization = "NYU_ORG"
        
        # Get organization key, use placeholder if not available
        org_key = secrets.get(self.organization, "placeholder")
        arg_org_key = secrets.get("ARG_ORG", "placeholder")
        
        self._openai_base = OpenAIBaseModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=org_key,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._openai_base_arg = OpenAIBaseModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=arg_org_key,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._openai_chat = OpenAIChatModel(
            frac_rate_limit=self.openai_fraction_rate_limit,
            organization=org_key,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._anthropic_chat = AnthropicChatModel(
            num_threads=self.anthropic_num_threads,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        self._ollama = OllamaModel(
            host=self.ollama_host,
            temperature=self.ollama_temperature,
            print_prompt_and_response=self.print_prompt_and_response,
        )
        Path("./prompt_history").mkdir(exist_ok=True)

    @staticmethod
    def _load_from_cache(save_file):
        if not os.path.exists(save_file):
            return None
        else:
            with open(save_file) as f:
                cache = json.load(f)
            return cache

    async def call_single(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        max_tokens: int,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        # is_valid: Callable[[str], bool] = lambda _: True,
        parse_fn: Optional[Callable] = None,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> str:
        assert n == 1, f"Expected a single response. {n} responses were requested."
        
        # Ensure parse_fn is not passed if it's the default lambda
        if parse_fn is not None and hasattr(parse_fn, '__name__') and parse_fn.__name__ == '<lambda>':
            parse_fn = None
            
        # Force single response parameters - remove n from kwargs to avoid conflict
        kwargs.pop('n', None)  # Remove n if it exists in kwargs
        
        responses = await self(
            model_ids,
            prompt,
            print_prompt_and_response=print_prompt_and_response,
            max_attempts_per_api_call=max_attempts_per_api_call,
            n=1,  # Force exactly 1 response
            num_candidates_per_completion=1,  # Force exactly 1 candidate
            parse_fn=parse_fn,
            insufficient_valids_behaviour=insufficient_valids_behaviour,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        if not responses or len(responses) == 0:
            raise ValueError("No responses received from the model")
        
        if len(responses) != 1:
            print(f"Debug: Expected 1 response, got {len(responses)}")
            print(f"Debug: Response types: {[type(r) for r in responses]}")
            if len(responses) > 0:
                print(f"Debug: First response: {responses[0]}")
        
        assert len(responses) == 1, f"Expected a single response, got {len(responses)}"
        
        # Extract completion from the response format
        response = responses[0]
        
        # Handle different response formats
        if isinstance(response, dict):
            if "response" in response:
                # Format: {"prompt": ..., "response": {"completion": "...", ...}}
                llm_response = response["response"]
                if isinstance(llm_response, dict) and "completion" in llm_response:
                    return llm_response["completion"]
                elif hasattr(llm_response, 'completion'):
                    return llm_response.completion
            elif "completion" in response:
                # Format: {"completion": "...", ...}
                return response["completion"]
        elif hasattr(response, 'completion'):
            # LLMResponse object
            return response.completion
        
        # Fallback - convert to string and hope for the best
        print(f"Warning: Unexpected response format: {type(response)}")
        print(f"Response content: {response}")
        return str(response)

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 50,
        num_candidates_per_completion: int = 1,
        parse_fn: Optional[Callable] = None,
        use_cache: bool = True,
        file_sem: asyncio.Semaphore = None,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion. n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter, then the remaining completions are returned up to a maximum of n.
            parse_fn: post-processing on the generated response
            save_path: cache path
            use_cache: whether to load from the cache or overwrite it
        """

        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]

        def model_id_to_class(model_id: str) -> ModelAPIProtocol:
            if model_id in ["gpt-4-base", "gpt-3.5-turbo-instruct"]:
                return (
                    self._openai_base_arg
                )  # NYU ARG is only org with access to this model
            elif model_id in BASE_MODELS:
                return self._openai_base
            elif model_id in GPT_CHAT_MODELS or "ft:gpt-3.5-turbo" in model_id:
                return self._openai_chat
            elif model_id in ANTHROPIC_MODELS:
                return self._anthropic_chat
            elif model_id in OLLAMA_MODELS or self._is_ollama_model(model_id):
                return self._ollama
            raise ValueError(f"Invalid model id: {model_id}")

        model_classes = [model_id_to_class(model_id) for model_id in model_ids]

        if len(set(str(type(x)) for x in model_classes)) != 1:
            raise ValueError("All model ids must be of the same type.")

        max_tokens = (
            kwargs.get("max_tokens") if kwargs.get("max_tokens") is not None else 2000
        )
        model_class = model_classes[0]
        if isinstance(model_class, AnthropicChatModel):
            kwargs["max_tokens_to_sample"] = max_tokens
        elif isinstance(model_class, OllamaModel):
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        
        # Check if current prompt has already been saved in the save file
        responses = None
        if use_cache and kwargs.get("save_path") is not None:
            try:
                responses = self._load_from_cache(kwargs.get("save_path"))
            except:
                logging.error(f"invalid cache data: {kwargs.get('save_path')}")

        if responses is None:
            # Ensure we have integers for the multiplication
            if callable(num_candidates_per_completion):
                num_candidates_int = 1
            else:
                try:
                    num_candidates_int = int(num_candidates_per_completion) if num_candidates_per_completion is not None else 1
                except (ValueError, TypeError):
                    num_candidates_int = 1
                
            if callable(n):
                n_int = 1
            else:
                try:
                    n_int = int(n) if n is not None else 1
                except (ValueError, TypeError):
                    n_int = 1
                
            # For Ollama, don't multiply - it handles one response at a time
            if isinstance(model_class, OllamaModel):
                num_candidates = n_int  # Just use n directly for Ollama
            else:
                # For other models, use the original logic
                if n_int == 1 and num_candidates_int == 1:
                    num_candidates = 1
                else:
                    num_candidates = num_candidates_int * n_int
            
            if isinstance(model_class, AnthropicChatModel):
                responses = list(
                    chain.from_iterable(
                        await asyncio.gather(
                            *[
                                model_class(
                                    model_ids,
                                    prompt,
                                    print_prompt_and_response,
                                    max_attempts_per_api_call,
                                    **kwargs,
                                )
                                for _ in range(num_candidates)
                            ]
                        )
                    )
                )
            elif isinstance(model_class, OllamaModel):
                # For Ollama, make exactly n_int calls
                responses = list(
                    chain.from_iterable(
                        await asyncio.gather(
                            *[
                                model_class(
                                    model_ids,
                                    prompt,
                                    print_prompt_and_response,
                                    max_attempts_per_api_call,
                                    **kwargs,
                                )
                                for _ in range(n_int)
                            ]
                        )
                    )
                )
            else:
                responses = await model_class(
                    model_ids,
                    prompt,
                    print_prompt_and_response,
                    max_attempts_per_api_call,
                    n=num_candidates,
                    **kwargs,
                )

        modified_responses = []
        for response in responses:
            # Handle the response based on its format
            if isinstance(response, dict) and "response" in response:
                # Already in the correct format from Ollama
                llm_response = response["response"]
                self.running_cost += llm_response.get("cost", 0)
                if kwargs.get("metadata") is not None:
                    response["metadata"] = kwargs.get("metadata")
                if parse_fn is not None and callable(parse_fn):
                    response = parse_fn(response)
                
                self.model_timings.setdefault(llm_response.get("model_id", "unknown"), []).append(
                    llm_response.get("api_duration", 0)
                )
                self.model_wait_times.setdefault(
                    llm_response.get("model_id", "unknown"), []
                ).append(
                    llm_response.get("duration", 0) - llm_response.get("api_duration", 0)
                )
                modified_responses.append(response)
            else:
                # Handle LLMResponse objects (from OpenAI/Anthropic)
                if hasattr(response, 'cost'):
                    self.running_cost += response.cost
                else:
                    self.running_cost += response.get("cost", 0) if isinstance(response, dict) else 0
                
                # Convert to expected format
                response_dict = {
                    "prompt": prompt,
                    "response": response.to_dict() if hasattr(response, 'to_dict') else response
                }
                
                if kwargs.get("metadata") is not None:
                    response_dict["metadata"] = kwargs.get("metadata")
                if parse_fn is not None and callable(parse_fn):
                    response_dict = parse_fn(response_dict)

                model_id = response.model_id if hasattr(response, 'model_id') else response_dict["response"].get("model_id", "unknown")
                api_duration = response.api_duration if hasattr(response, 'api_duration') else response_dict["response"].get("api_duration", 0)
                duration = response.duration if hasattr(response, 'duration') else response_dict["response"].get("duration", 0)
                
                self.model_timings.setdefault(model_id, []).append(api_duration)
                self.model_wait_times.setdefault(model_id, []).append(duration - api_duration)
                modified_responses.append(response_dict)

        if kwargs.get("save_path") is not None:
            if file_sem is not None:
                async with file_sem:
                    with open(kwargs.get("save_path"), "w") as f:
                        json.dump(modified_responses, f, indent=2)
            else:
                with open(kwargs.get("save_path"), "w") as f:
                    json.dump(modified_responses, f, indent=2)
        
        # Ensure n_int is defined for the return statement
        if 'n_int' not in locals():
            try:
                n_int = int(n) if n is not None else 1
            except (ValueError, TypeError):
                n_int = 1
        
        return modified_responses[:n_int]

    def _is_ollama_model(self, model_id: str) -> bool:
        """Check if a model_id looks like an Ollama model."""
        # Ollama models typically have format: model_name:tag or just model_name
        # They don't contain "gpt", "claude", or other API provider prefixes
        if any(prefix in model_id.lower() for prefix in ["gpt", "claude", "text-", "ft:"]):
            return False
        
        # If it contains a colon, it's likely an Ollama model with a tag
        if ":" in model_id:
            return True
            
        # Check if it's a common model name that could be an Ollama model
        common_ollama_names = [
            "llama", "mistral", "mixtral", "qwen", "gemma", "phi", "vicuna", 
            "orca", "wizard", "neural", "starling", "codellama", "deepseek"
        ]
        return any(name in model_id.lower() for name in common_ollama_names)

    def reset_cost(self):
        self.running_cost = 0

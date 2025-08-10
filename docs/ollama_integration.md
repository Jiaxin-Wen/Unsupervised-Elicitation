# Ollama Integration Guide

This guide explains how to use Ollama models with the Unsupervised-Elicitation project instead of vLLM.

## What is Ollama?

Ollama is a tool for running large language models locally on your machine. It provides:
- Easy model management (pull, run, delete models)
- Local inference (no API keys needed)
- Support for many popular models (Llama, Mistral, CodeLlama, etc.)
- Simple HTTP API

## Installation

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai) or run:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

The requirements now include:
- `ollama` - Python client for Ollama
- `langchain-ollama` - LangChain integration
- `langchain-core` - Core LangChain functionality

### 3. Start Ollama Service

```bash
ollama serve
```

### 4. Pull Models

```bash
# Pull some recommended models
ollama pull llama3.2:latest      # Latest Llama model
ollama pull codellama:latest     # Code generation
ollama pull mistral:latest       # Fast general purpose
```

Or use the setup script:
```bash
python setup_ollama.py
```

## Usage

### Basic Usage

```python
from core.llm_api import ModelAPI

# Initialize with Ollama settings
model_api = ModelAPI(
    ollama_host="http://localhost:11434",  # Default
    ollama_temperature=0.1,
    print_prompt_and_response=True
)

# Simple text completion
response = await model_api.call_single(
    model_ids="llama3.2:latest",
    prompt="What is machine learning?",
    max_tokens=100
)
print(response)
```

### Chat-Style Conversation

```python
responses = await model_api(
    model_ids=["llama3.2:latest"],
    prompt=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python decorators."}
    ],
    max_tokens=150,
    n=1
)

for response in responses:
    print(response['response']['completion'])
```

### Using Different Models

```python
# Code generation with CodeLlama
code_response = await model_api.call_single(
    model_ids="codellama:latest",
    prompt="Write a Python function to sort a list:",
    max_tokens=200
)

# Fast responses with Mistral
quick_response = await model_api.call_single(
    model_ids="mistral:latest", 
    prompt="Summarize: What is AI?",
    max_tokens=50
)
```

## Available Models

The system automatically detects Ollama models. Popular options:

### General Purpose
- `llama3.2:latest` - Latest Llama model
- `llama3.1:8b` - Llama 3.1 8B parameters
- `mistral:latest` - Fast Mistral model
- `qwen2.5:7b` - Multilingual Qwen model

### Code Specialized
- `codellama:latest` - Code generation
- `deepseek-coder:latest` - Code understanding
- `codeqwen:latest` - Code and text

### Small/Fast Models
- `llama3.2:3b` - Smaller Llama model
- `phi3:latest` - Microsoft Phi-3
- `gemma2:2b` - Small Gemma model

Browse all available models at [ollama.ai/library](https://ollama.ai/library)

## Configuration

### ModelAPI Parameters

```python
model_api = ModelAPI(
    ollama_host="http://localhost:11434",  # Ollama server URL
    ollama_temperature=0.0,                # Default temperature
    print_prompt_and_response=False        # Debug output
)
```

### Model Detection

The system automatically detects Ollama models by:
1. Checking if model_id is in the predefined `OLLAMA_MODELS` set
2. Using heuristics (contains ":", common model names like "llama", "mistral")
3. Excluding API provider patterns ("gpt", "claude", etc.)

## Migration from vLLM

### Before (vLLM)
```python
# vLLM required complex setup and model loading
model_api = ModelAPI()
response = await model_api("meta-llama/Llama-3-8b", prompt)
```

### After (Ollama)
```python
# Ollama models are pulled once and run locally
model_api = ModelAPI()
response = await model_api("llama3.2:latest", prompt)
```

### Key Differences

| Feature | vLLM | Ollama |
|---------|------|---------|
| Setup | Complex, requires model files | Simple, pull from registry |
| Cost | Free (local) | Free (local) |
| Models | Manual download/setup | One-command pull |
| Memory | Loads model in memory | Manages automatically |
| API | Python API | HTTP + Python client |

## Troubleshooting

### Common Issues

1. **"Ollama not available"**
   ```bash
   pip install ollama langchain-ollama langchain-core
   ```

2. **"Could not connect to Ollama"**
   ```bash
   ollama serve  # Start the service
   ```

3. **"Model not found"**
   ```bash
   ollama pull llama3.2:latest
   ```

4. **Slow responses**
   - Use smaller models (3b instead of 70b)
   - Reduce max_tokens
   - Check system resources

### Debug Mode

Enable debug output to see what's happening:
```python
model_api = ModelAPI(print_prompt_and_response=True)
```

### Check Available Models

```bash
ollama list  # List pulled models
ollama ps    # List running models
```

## Examples

Run the example script:
```bash
python examples/ollama_example.py
```

Or check the demo in the main module:
```bash
python -c "from core.llm_api.llm import demo; import asyncio; asyncio.run(demo())"
```

## Performance Tips

1. **Model Size**: Larger models (70b) are more capable but slower
2. **Context Length**: Shorter prompts = faster responses
3. **Temperature**: Lower values (0.1) = more deterministic, faster
4. **Caching**: Ollama automatically caches models in memory
5. **Hardware**: More RAM = can run larger models

## Integration with Existing Code

The Ollama integration is designed to be a drop-in replacement. Existing code using the ModelAPI should work with minimal changes:

```python
# This works with OpenAI, Anthropic, AND Ollama models
model_api = ModelAPI()

# Will automatically use the right backend
response = await model_api.call_single("gpt-4", prompt)        # OpenAI
response = await model_api.call_single("claude-3", prompt)     # Anthropic  
response = await model_api.call_single("llama3.2", prompt)    # Ollama
```

The system automatically detects which backend to use based on the model name.

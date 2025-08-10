#!/usr/bin/env python3
"""
Simple example of using Ollama models with the ModelAPI.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_api import ModelAPI


async def main():
    print("Ollama Integration Example")
    print("=" * 40)
    
    # Initialize ModelAPI with Ollama settings
    model_api = ModelAPI(
        ollama_host="http://localhost:11434",
        ollama_temperature=0.1
    )
    
    # Example 1: Simple arithmetic
    print("1. Testing simple arithmetic:")
    model_name = "llama3.2:latest"  # Change this to your preferred model
    prompt = [
        {"role": "user", "content": "What is 2+2? Answer with just the number."}
    ]
    
    try:
        print(f"Testing model: {model_name}")
        print("Sending prompt...")
        
        # Get single response
        response = await model_api.call_single(
            model_name,
            prompt,
            max_tokens=10
        )
        
        print(f"Response: {response}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Trying with a different model...\n")
        
        # Fallback to a different model
        try:
            response = await model_api.call_single(
                "llama3.2:1b",
                prompt,
                max_tokens=10
            )
            print(f"Response (llama3.2:1b): {response}\n")
        except Exception as e2:
            print(f"Error with fallback: {e2}\n")
    
    # Example 2: Complex reasoning (ICM-style prompt)
    print("2. Testing complex reasoning (ICM-style):")
    complex_prompt = [
        {"role": "user", "content": """
Question: A baker has 15 cookies. He gives 3 cookies to each customer. How many customers can he serve?
Claim: The baker can serve 5 customers.
I think this claim is """}
    ]
    
    try:
        complex_response = await model_api.call_single(
            model_name,
            complex_prompt,
            max_tokens=5
        )
        print(f"Complex response: {complex_response}\n")
        
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Code generation
    print("3. Testing code generation:")
    try:
        code_response = await model_api.call_single(
            "codellama:latest",
            "Write a Python function to calculate the factorial of a number:",
            max_tokens=200
        )
        print(f"Generated code:\n{code_response}\n")
    except Exception as e:
        print(f"Error (trying with llama3.2): {e}")
        # Fallback to llama3.2 if codellama is not available
        try:
            response = await model_api.call_single(
                "llama3.2:latest",
                "Write a Python function to calculate the factorial of a number:",
                max_tokens=200
            )
            print(f"Generated code:\n{response}\n")
        except Exception as e2:
            print(f"Error: {e2}\n")
    
    # Example 4: Multiple different models (if available)
    print("4. Trying different models:")
    models_to_try = [
        "llama3.2:latest",
        "llama3.1:latest", 
        "mistral:latest",
        "qwen2.5:latest"
    ]
    
    for model in models_to_try:
        try:
            response = await model_api.call_single(
                model,
                "Hi! What's your name?",
                max_tokens=50
            )
            print(f"{model}: {response}")
        except Exception as e:
            print(f"{model}: Not available ({e})")
    
    print(f"\nTotal cost: ${model_api.running_cost:.4f}")
    print("\n=== Example completed ===")
    print("Tip: You can pull more models with 'ollama pull <model_name>'")
    print("Available models: https://ollama.ai/library")
    
    print("\nTroubleshooting tips:")
    print("1. Make sure Ollama is running: ollama serve")
    print("2. Make sure models are available: ollama list")
    print("3. Pull models: ollama pull llama3.2:latest")


if __name__ == "__main__":
    asyncio.run(main())

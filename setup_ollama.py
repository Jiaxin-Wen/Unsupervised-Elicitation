#!/usr/bin/env python3
"""
Setup script for Ollama integration.

This script helps set up the Ollama environment and pulls recommended models.
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description if description else cmd}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Ollama is not installed or not in PATH")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        return False


def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama service is running")
            return True
        else:
            print("✗ Ollama service is not running")
            return False
    except Exception:
        print("✗ Cannot check Ollama service status")
        return False


def list_available_models():
    """List currently available models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Currently available models:")
            print(result.stdout)
            return result.stdout
        else:
            print("Could not list models")
            return ""
    except Exception as e:
        print(f"Error listing models: {e}")
        return ""


def create_secrets_file():
    """Create a minimal SECRETS file for Ollama usage."""
    secrets_path = Path("SECRETS")
    
    if secrets_path.exists():
        print("✓ SECRETS file already exists")
        return True
    
    print("Creating minimal SECRETS file for Ollama usage...")
    secrets_content = """API_KEY=placeholder
LLAMA_API_BASE=http://localhost:11434
NYU_ORG=placeholder
ARG_ORG=placeholder
ANTHROPIC_API_KEY=placeholder
MISTRAL_API_KEY=placeholder
REPLICATE_API_KEY=placeholder
"""
    
    try:
        with open(secrets_path, 'w') as f:
            f.write(secrets_content)
        print("✓ Created SECRETS file")
        return True
    except Exception as e:
        print(f"✗ Failed to create SECRETS file: {e}")
        return False


def pull_model(model_name):
    """Pull a specific model."""
    return run_command(f"ollama pull {model_name}", f"Pulling model: {model_name}")


def main():
    print("Ollama Setup Script for Unsupervised-Elicitation")
    print("=" * 60)
    
    # Create SECRETS file first
    create_secrets_file()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\nPlease install Ollama first:")
        print("- Visit: https://ollama.ai/")
        print("- Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("\nStarting Ollama service...")
        print("Run this in a separate terminal: ollama serve")
        input("Press Enter after starting Ollama service...")
        
        if not check_ollama_running():
            print("Ollama service is still not running. Please start it manually.")
            return
    
    # List current models
    print("\n" + "="*60)
    print("CURRENT MODELS")
    print("="*60)
    list_available_models()
    
    # Recommended models for this project
    recommended_models = [
        ("llama3.2:latest", "Latest Llama 3.2 model - good general purpose"),
        ("llama3.1:8b", "Llama 3.1 8B - balanced performance/speed"),
        ("codellama:latest", "Code generation specialist"),
        ("mistral:latest", "Fast and efficient general model"),
        ("qwen2.5:7b", "Good multilingual capabilities"),
    ]
    
    print("\n" + "="*60)
    print("RECOMMENDED MODELS")
    print("="*60)
    
    for model, description in recommended_models:
        print(f"{model:<20} - {description}")
    
    print("\nWould you like to pull some recommended models?")
    print("Note: Models can be large (several GB each)")
    
    # Ask user which models to pull
    for model, description in recommended_models:
        while True:
            response = input(f"\nPull {model}? ({description}) [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                success = pull_model(model)
                if success:
                    print(f"✓ Successfully pulled {model}")
                else:
                    print(f"✗ Failed to pull {model}")
                break
            elif response in ['n', 'no', '']:
                print(f"Skipping {model}")
                break
            else:
                print("Please enter 'y' or 'n'")
    
    # Final status
    print("\n" + "="*60)
    print("FINAL STATUS")
    print("="*60)
    list_available_models()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Install Python dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run the verification script:")
    print("   python verify_ollama_integration.py")
    print()
    print("3. Run the example script:")
    print("   python examples/ollama_example.py")
    print()
    print("4. Or use in your code:")
    print("   from core.llm_api import ModelAPI")
    print("   model_api = ModelAPI()")
    print("   response = await model_api.call_single('llama3.2:latest', 'Hello!', max_tokens=100)")
    print()
    print("5. Browse more models at: https://ollama.ai/library")


if __name__ == "__main__":
    main()

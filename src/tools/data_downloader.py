import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any

import requests
from datasets import load_dataset

from src.tools.path_utils import get_root_directory


def ensure_data_directory():
    """Ensure the data directory exists."""
    data_dir = get_root_directory() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_gsm8k_data(data_dir: Path) -> None:
    """Download and prepare GSM8K dataset."""
    print("Downloading GSM8K dataset...")
    
    try:
        # Load GSM8K dataset from HuggingFace
        dataset = load_dataset("gsm8k", "main")
        train_data = dataset["train"]
        
        # Convert to the expected format
        formatted_data = []
        for i, item in enumerate(train_data):
            # Extract question and answer
            question = item["question"]
            answer = item["answer"]
            
            # Create multiple choice variants or claims
            formatted_item = {
                "consistency_id": i,
                "question": question,
                "choice": answer,
                "answer": answer,
                "label": 1,  # Correct answer
                "vanilla_label": 1,
                "uid": i
            }
            formatted_data.append(formatted_item)
        
        # Save to JSON file
        output_file = data_dir / "train_gsm8k.json"
        with open(output_file, "w") as f:
            json.dump(formatted_data, f, indent=2)
        
        print(f"GSM8K data saved to {output_file}")
        
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")
        # Create minimal sample data as fallback
        create_sample_gsm8k_data(data_dir)


def create_sample_gsm8k_data(data_dir: Path) -> None:
    """Create sample GSM8K data for testing."""
    print("Creating sample GSM8K data...")
    
    sample_data = [
        {
            "consistency_id": 0,
            "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "choice": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She sells them for $2 each, so she makes 9 × $2 = $18.",
            "answer": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left to sell. She sells them for $2 each, so she makes 9 × $2 = $18.",
            "label": 1,
            "vanilla_label": 1,
            "uid": 0
        },
        {
            "consistency_id": 1,
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts are needed?",
            "choice": "A robe takes 2 bolts of blue fiber and half that much white fiber. Half of 2 is 1, so it takes 1 bolt of white fiber. In total, 2 + 1 = 3 bolts are needed.",
            "answer": "A robe takes 2 bolts of blue fiber and half that much white fiber. Half of 2 is 1, so it takes 1 bolt of white fiber. In total, 2 + 1 = 3 bolts are needed.",
            "label": 1,
            "vanilla_label": 1,
            "uid": 1
        }
    ]
    
    # Extend sample data to have more examples
    for i in range(2, 20):
        sample_data.append({
            "consistency_id": i,
            "question": f"Sample math question {i}?",
            "choice": f"Sample answer {i}",
            "answer": f"Sample answer {i}",
            "label": random.choice([0, 1]),
            "vanilla_label": random.choice([0, 1]),
            "uid": i
        })
    
    output_file = data_dir / "train_gsm8k.json"
    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample GSM8K data saved to {output_file}")


def download_alpaca_data(data_dir: Path) -> None:
    """Download and prepare Alpaca dataset."""
    print("Downloading Alpaca dataset...")
    
    try:
        # Try to download from the original source
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        response = requests.get(url)
        
        if response.status_code == 200:
            alpaca_data = response.json()
            
            # Convert to expected format
            formatted_data = []
            for i, item in enumerate(alpaca_data[:100]):  # Limit to first 100 for testing
                formatted_item = {
                    "consistency_id": i,
                    "question": item["instruction"],
                    "choice": item["output"],
                    "choice_2": item["output"][::-1],  # Reversed as alternative
                    "label": random.choice([0, 1]),
                    "vanilla_label": random.choice([0, 1]),
                    "uid": i
                }
                formatted_data.append(formatted_item)
            
            output_file = data_dir / "train_alpaca.json"
            with open(output_file, "w") as f:
                json.dump(formatted_data, f, indent=2)
            
            print(f"Alpaca data saved to {output_file}")
        else:
            raise Exception(f"Failed to download: status {response.status_code}")
            
    except Exception as e:
        print(f"Error downloading Alpaca: {e}")
        create_sample_alpaca_data(data_dir)


def create_sample_alpaca_data(data_dir: Path) -> None:
    """Create sample Alpaca data for testing."""
    print("Creating sample Alpaca data...")
    
    sample_data = []
    for i in range(20):
        sample_data.append({
            "consistency_id": i,
            "question": f"Sample instruction {i}",
            "choice": f"Sample response A {i}",
            "choice_2": f"Sample response B {i}",
            "label": random.choice([0, 1]),
            "vanilla_label": random.choice([0, 1]),
            "uid": i
        })
    
    output_file = data_dir / "train_alpaca.json"
    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample Alpaca data saved to {output_file}")


def download_truthfulqa_data(data_dir: Path) -> None:
    """Download and prepare TruthfulQA dataset."""
    print("Downloading TruthfulQA dataset...")
    
    try:
        dataset = load_dataset("truthful_qa", "generation")
        validation_data = dataset["validation"]
        
        formatted_data = []
        for i, item in enumerate(validation_data):
            formatted_item = {
                "consistency_id": i,
                "question": item["question"],
                "choice": item["best_answer"] if "best_answer" in item else "Sample answer",
                "label": 1,
                "vanilla_label": 1,
                "uid": i
            }
            formatted_data.append(formatted_item)
        
        output_file = data_dir / "train_truthfulqa.json"
        with open(output_file, "w") as f:
            json.dump(formatted_data, f, indent=2)
        
        print(f"TruthfulQA data saved to {output_file}")
        
    except Exception as e:
        print(f"Error downloading TruthfulQA: {e}")
        create_sample_truthfulqa_data(data_dir)


def create_sample_truthfulqa_data(data_dir: Path) -> None:
    """Create sample TruthfulQA data for testing."""
    print("Creating sample TruthfulQA data...")
    
    sample_data = []
    for i in range(20):
        sample_data.append({
            "consistency_id": i,
            "question": f"Sample truthfulness question {i}?",
            "choice": f"Sample truthful answer {i}",
            "label": random.choice([0, 1]),
            "vanilla_label": random.choice([0, 1]),
            "uid": i
        })
    
    output_file = data_dir / "train_truthfulqa.json"
    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample TruthfulQA data saved to {output_file}")


def create_sample_truthfulqa_preference_data(data_dir: Path) -> None:
    """Create sample TruthfulQA preference data for testing."""
    print("Creating sample TruthfulQA preference data...")
    
    sample_data = []
    for i in range(20):
        sample_data.append({
            "consistency_id": i,
            "question": f"Sample preference question {i}?",
            "choice": f"Sample answer A {i}",
            "choice_2": f"Sample answer B {i}",
            "label": random.choice([0, 1]),
            "vanilla_label": random.choice([0, 1]),
            "uid": i
        })
    
    output_file = data_dir / "train_truthfulqa_preference.json"
    with open(output_file, "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample TruthfulQA preference data saved to {output_file}")


def download_all_datasets():
    """Download all required datasets."""
    data_dir = ensure_data_directory()
    
    print("Downloading datasets...")
    
    # Download each dataset
    download_gsm8k_data(data_dir)
    download_alpaca_data(data_dir)
    download_truthfulqa_data(data_dir)
    create_sample_truthfulqa_preference_data(data_dir)
    
    print("All datasets downloaded/created successfully!")


if __name__ == "__main__":
    download_all_datasets()

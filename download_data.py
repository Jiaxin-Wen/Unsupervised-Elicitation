#!/usr/bin/env python3
"""
Convenience script to download all required datasets for the ICM experiments.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.tools.data_downloader import download_all_datasets

if __name__ == "__main__":
    print("Downloading all required datasets...")
    download_all_datasets()
    print("\nDone! You can now run the ICM experiments.")

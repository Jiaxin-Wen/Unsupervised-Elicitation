import asyncio
import json
import logging
import os
import time
from functools import lru_cache, wraps
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import replicate
import typer
import yaml
from tenacity import retry, retry_if_result, stop_after_attempt

typer.main.get_command_name = lambda name: name

LOGGER = logging.getLogger(__name__)
SEPARATOR = "---------------------------------------------\n\n"
SEPARATOR_CONVERSATIONAL_TURNS = "=============================================\n\n"
PROMPT_HISTORY = "prompt_history"
SECRETS_FILE_PATH = Path(__file__).parent.parent / "SECRETS"

LOGGING_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def setup_environment(
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logger_level: str = "info",
    openai_tag: str = "API_KEY",
    mistral_tag: str = "MISTRAL_API_KEY",
    replicate_tag: str = "REPLICATE_API_KEY",
    organization: str = None,
):
    setup_logging(logger_level)
    load_secrets(
        SECRETS_FILE_PATH,
        anthropic_tag,
        logger_level,
        openai_tag,
        mistral_tag,
        replicate_tag,
        organization,
    )


def setup_logging(level_str):
    level = LOGGING_LEVELS.get(
        level_str.lower(), logging.INFO
    )  # default to INFO if level_str is not found
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Disable logging from noisy libraries
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    logging.getLogger("anthropic").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    LOGGER.info(f"Logging level set to {level_str}")


def load_secrets(
    file_path=SECRETS_FILE_PATH,
    anthropic_tag: str = "ANTHROPIC_API_KEY",
    logger_level: str = "info",
    openai_tag: str = "API_KEY",
    mistral_tag: str = "MISTRAL_API_KEY",
    replicate_tag: str = "REPLICATE_API_KEY",
    organization: str = None,
):
    secrets = {}
    
    # Check if secrets file exists
    if not os.path.exists(file_path):
        LOGGER.warning(f"Secrets file not found at {file_path}. Using default values for Ollama-only setup.")
        # Provide minimal defaults for Ollama usage
        secrets = {
            openai_tag: "placeholder",
            "LLAMA_API_BASE": "http://localhost:11434",
            "NYU_ORG": "placeholder",
            "ARG_ORG": "placeholder",
            anthropic_tag: "placeholder",
            mistral_tag: "placeholder",
            replicate_tag: "placeholder",
        }
    else:
        with open(file_path) as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split("=", 1)
                    secrets[key] = value

    # Only set OpenAI key if it's not a placeholder
    if secrets.get(openai_tag, "placeholder") != "placeholder":
        openai.api_key = secrets[openai_tag]
    
    if secrets.get('LLAMA_API_BASE'):
        os.environ['LLAMA_API_BASE'] = secrets['LLAMA_API_BASE']
    
    # Only set other environment variables if they exist and aren't placeholders
    for env_key, secret_key in [
        ("ANTHROPIC_API_KEY", anthropic_tag),
        ("MISTRAL_API_KEY", mistral_tag), 
        ("REPLICATE_API_KEY", replicate_tag)
    ]:
        if secrets.get(secret_key) and secrets[secret_key] != "placeholder":
            os.environ[env_key] = secrets[secret_key]

    if organization is not None and secrets.get(organization):
        openai.organization = secrets[organization]
    if secrets.get("API_BASE") is not None:
        openai.api_base = secrets['API_BASE']
    return secrets


def load_yaml(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


@lru_cache(maxsize=8)
def load_yaml_cached(file_path):
    with open(file_path) as f:
        content = yaml.safe_load(f)
    return content


def save_yaml(file_path, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_obj = json.loads(line)
            data.append(json_obj)
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def delete_old_prompt_files(
    path: str = PROMPT_HISTORY, max_age_minutes: int = 60, keep_recent: int = 50
):
    """
    Delete all files in the folder that:
    - Are more than max_age_minutes old
    - AND are not one of the y_most_recent files, delete it
    """
    if not os.path.exists(path):
        return

    # Get all files in the folder with their full paths and creation times
    files = [
        {
            "path": os.path.join(path, filename),
            "ctime": os.path.getctime(os.path.join(path, filename)),
        }
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename))
    ]

    # Sort files by creation time
    files.sort(key=lambda f: f["ctime"], reverse=True)

    # Current time in seconds since epoch
    now = time.time()

    deleted_count = 0
    for index, file_info in enumerate(files):
        # File age in minutes
        age_minutes = (now - file_info["ctime"]) / 60

        # If file is older than x_minutes and is not one of the y_most_recent files, delete it
        if age_minutes > max_age_minutes and index >= keep_recent:
            os.remove(file_info["path"])
            deleted_count += 1

    if deleted_count > 0:
        print(f"Deleted {deleted_count} old prompt files")


def typer_async(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # No event loop running
            loop = None

        if loop is None:
            return asyncio.run(f(*args, **kwargs))
        else:
            return f(*args, **kwargs)  # Return coroutine to be awaited

    return wrapper


@retry(
    stop=stop_after_attempt(16),
    retry=retry_if_result(lambda result: result is not True),
)
def function_with_retry(function, *args, **kwargs):
    return function(*args, **kwargs)


@retry(
    stop=stop_after_attempt(16),
    retry=retry_if_result(lambda result: result is not True),
)
async def async_function_with_retry(function, *args, **kwargs):
    return await function(*args, **kwargs)


def log_model_timings(api_handler, save_location="./model_timings.png"):
    if len(api_handler.model_timings) > 0:
        plt.figure(figsize=(10, 6))
        for model in api_handler.model_timings:
            timings = np.array(api_handler.model_timings[model])
            wait_times = np.array(api_handler.model_wait_times[model])
            LOGGER.info(
                f"{model}: response {timings.mean():.3f}, waiting {wait_times.mean():.3f} (max {wait_times.max():.3f}, min {wait_times.min():.3f})"
            )
            plt.plot(
                timings, label=f"{model} - Response Time", linestyle="-", linewidth=2
            )
            plt.plot(
                wait_times, label=f"{model} - Waiting Time", linestyle="--", linewidth=2
            )
        plt.legend()
        plt.title("Model Performance: Response and Waiting Times")
        plt.xlabel("Sample Number")
        plt.ylabel("Time (seconds)")
        plt.savefig(save_location, dpi=300)
        plt.close()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

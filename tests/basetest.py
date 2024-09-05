import pytest
from unifai import UnifAIClient

from os import getenv
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

def base_test_all_providers(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
    (
        "anthropic", 
        {"api_key": ANTHROPIC_API_KEY},
        {}
        ),
    (
        "openai", 
        {"api_key": OPENAI_API_KEY},
        {}
    ),
    (
        "ollama", 
        {"host": "http://librem-2.local:11434"},
        {"keep_alive": "10m", 'model': 'llama3.1-8b-num_ctx-8192:latest'}
    ),
    ])(func)


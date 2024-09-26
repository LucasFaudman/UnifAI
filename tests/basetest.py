import pytest
from unifai import UnifAIClient

from os import getenv
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = getenv("_ANTHROPIC_API_KEY")
GOOGLE_API_KEY = getenv("_GOOGLE_API_KEY")
OPENAI_API_KEY = getenv("_OPENAI_API_KEY")

PROVIDER_DEFAULTS = {
    # "provider": (provider, client_kwargs, func_kwargs)
    "anthropic": (
        "anthropic", 
        {"api_key": ANTHROPIC_API_KEY},
        {}
    ),
    "google": (
        "google",
        {"api_key": GOOGLE_API_KEY},
        {}   
    ),
    "openai": (
        "openai", 
        {"api_key": OPENAI_API_KEY},
        {}
    ), 
    "ollama": (
        "ollama", 
        {"host": "http://librem-2.local:11434"},
        {"keep_alive": "10m", 'model': 'llama3.1-8b-num_ctx-8192:latest'}
    ),
}

def base_test_all_providers(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        *list(PROVIDER_DEFAULTS.values())[:-1]
    ])(func)

def base_test_no_anthropic(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for provider, defaults in PROVIDER_DEFAULTS.items() if provider != "anthropic"
    ])(func)

def base_test_no_google(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for provider, defaults in PROVIDER_DEFAULTS.items() if provider != "google"
    ])(func)

def base_test_no_openai(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for provider, defaults in PROVIDER_DEFAULTS.items() if provider != "openai"
    ])(func)

def base_test_no_ollama(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for provider, defaults in PROVIDER_DEFAULTS.items() if provider != "ollama"
    ])(func)
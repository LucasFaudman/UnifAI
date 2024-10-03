import pytest
from unifai import UnifAIClient

from os import getenv
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = getenv("_ANTHROPIC_API_KEY")
GOOGLE_API_KEY = getenv("_GOOGLE_API_KEY")
OPENAI_API_KEY = getenv("_OPENAI_API_KEY")
PINECONE_API_KEY = getenv("_PINECONE_API_KEY")
COHERE_API_KEY = getenv("_COHERE_API_KEY")

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

    "chroma": (
        "chroma",
        {"persist_directory": "/Users/lucasfaudman/Documents/UnifAI/scratch/test_embeddings",
         "is_persistent": False
         },
        {}
    ),

    "pinecone": (
        "pinecone",
        {"api_key": PINECONE_API_KEY},
        {}
    ),   

    "cohere": (
        "cohere",
        {"args": (COHERE_API_KEY,)},
        {}
    ),       

}

AI_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["anthropic"],
    PROVIDER_DEFAULTS["google"],
    PROVIDER_DEFAULTS["openai"],
    PROVIDER_DEFAULTS["ollama"]
]

VECTOR_DB_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["chroma"],
    PROVIDER_DEFAULTS["pinecone"]
]

def base_test_all_providers(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", AI_PROVIDER_DEFAULTS[:])(func)

def base_test_no_anthropic(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for defaults in AI_PROVIDER_DEFAULTS if defaults[0] != "anthropic"
    ])(func)

def base_test_no_google(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for defaults in AI_PROVIDER_DEFAULTS if defaults[0] != "google"
    ])(func)

def base_test_no_openai(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for defaults in AI_PROVIDER_DEFAULTS if defaults[0] != "openai"
    ])(func)

def base_test_no_ollama(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for defaults in AI_PROVIDER_DEFAULTS if defaults[0] != "ollama"
    ])(func)


def base_test_all_db_providers(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", VECTOR_DB_PROVIDER_DEFAULTS)(func)

def base_test_db_no_chroma(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for defaults in VECTOR_DB_PROVIDER_DEFAULTS if defaults[0] != "chroma"
    ])(func)

def base_test_db_no_pinecone(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
        defaults for defaults in VECTOR_DB_PROVIDER_DEFAULTS if defaults[0] != "pinecone"
    ])(func)
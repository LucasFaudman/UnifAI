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
NVIDIA_API_KEY = getenv("_NVIDIA_API_KEY")

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
    "nvidia": (
        "nvidia", 
        {"api_key": NVIDIA_API_KEY},
        {}
    ),     
    "ollama": (
        "ollama", 
        {"host": "http://librem-2.local:11434"},
        {"keep_alive": "10m", 'model': 'llama3.1-8b-num_ctx-8192:latest'}
    ),

    "chroma": (
        "chroma",
        {
            "persist_directory": "/Users/lucasfaudman/Documents/UnifAI/scratch/test_embeddings",         
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
        {"api_key": COHERE_API_KEY},
        {}
    ),  

    "rank_bm25": (
        "rank_bm25",
        {},
        {}
    ),  
    "sentence_transformers": (
        "sentence_transformers",
        {},
        {}
    ),  

}

LLM_PROVIDER_DEFAULTS = [
    # PROVIDER_DEFAULTS["anthropic"],
    # PROVIDER_DEFAULTS["google"],
    # PROVIDER_DEFAULTS["openai"],
    # PROVIDER_DEFAULTS["ollama"],
    # PROVIDER_DEFAULTS["cohere"],
    PROVIDER_DEFAULTS["nvidia"]
]
LLM_PROVIDERS = [provider[0] for provider in LLM_PROVIDER_DEFAULTS]

EMBEDDING_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["google"],
    PROVIDER_DEFAULTS["openai"],
    # PROVIDER_DEFAULTS["ollama"],
    # PROVIDER_DEFAULTS["chroma"],
    # PROVIDER_DEFAULTS["pinecone"],
    PROVIDER_DEFAULTS["cohere"],
    PROVIDER_DEFAULTS["sentence_transformers"],
    PROVIDER_DEFAULTS["nvidia"]    
]
EMBEDDING_PROVIDERS = [provider[0] for provider in EMBEDDING_PROVIDER_DEFAULTS]


VECTOR_DB_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["chroma"],
    PROVIDER_DEFAULTS["pinecone"]
]
VECTOR_DB_PROVIDERS = [provider[0] for provider in VECTOR_DB_PROVIDER_DEFAULTS]

RERANKER_PROVIDER_DEFAULTS = [
    # PROVIDER_DEFAULTS["ollama"]
    PROVIDER_DEFAULTS["cohere"],
    PROVIDER_DEFAULTS["rank_bm25"],
    PROVIDER_DEFAULTS["sentence_transformers"],
    PROVIDER_DEFAULTS["nvidia"]    
]
RERANKER_PROVIDERS = [provider[0] for provider in RERANKER_PROVIDER_DEFAULTS]

def decorator_with_params(param1, param2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Decorator parameters: {param1}, {param2}")
            print("Before function execution")
            result = func(*args, **kwargs)
            print("After function execution")
            return result
        return wrapper
    return decorator

def base_test(*providers, exclude=[]):
    def decorator(func):
        return pytest.mark.parametrize(
            "provider, client_kwargs, func_kwargs", 
            [PROVIDER_DEFAULTS[provider] for provider in providers if provider not in exclude]
        )(func)
    return decorator

# parameterized test decorators


# LLM test decorators
def base_test_llms_all(func):
    return base_test(*LLM_PROVIDERS)(func)

# def base_test_llms_no_anthropic(func):
#     return base_test(*LLM_PROVIDERS, exclude=["anthropic"])(func)

# def base_test_llms_no_google(func):
#     return base_test(*LLM_PROVIDERS, exclude=["google"])(func)

# def base_test_llms_no_openai(func):
#     return base_test(*LLM_PROVIDERS, exclude=["openai"])(func)

# def base_test_llms_no_ollama(func):
#     return base_test(*LLM_PROVIDERS, exclude=["ollama"])(func)

# def base_test_llms_no_cohere(func):
#     return base_test(*LLM_PROVIDERS, exclude=["cohere"])(func)


# Embedding test decorators
def base_test_embeddings_all(func):
    return base_test(*EMBEDDING_PROVIDERS)(func)

# Vector DB test decorators
def base_test_vector_dbs_all(func):
    return base_test(*VECTOR_DB_PROVIDERS)(func)

# Reranker test decorators
def base_test_rerankers_all(func):
    return base_test(*RERANKER_PROVIDERS)(func)


# def base_test_all_llms(func):
#     return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", LLM_PROVIDER_DEFAULTS[:])(func)

# def base_test_no_anthropic(func):
#     return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
#         defaults for defaults in LLM_PROVIDER_DEFAULTS if defaults[0] != "anthropic"
#     ])(func)

# def base_test_no_google(func):
#     return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
#         defaults for defaults in LLM_PROVIDER_DEFAULTS if defaults[0] != "google"
#     ])(func)

# def base_test_no_openai(func):
#     return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
#         defaults for defaults in LLM_PROVIDER_DEFAULTS if defaults[0] != "openai"
#     ])(func)

# def base_test_no_ollama(func):
#     return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
#         defaults for defaults in LLM_PROVIDER_DEFAULTS if defaults[0] != "ollama"
#     ])(func)





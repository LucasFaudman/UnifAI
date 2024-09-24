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
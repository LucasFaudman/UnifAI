from os import getenv
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = getenv("_ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = getenv("_GOOGLE_API_KEY", "")
OPENAI_API_KEY = getenv("_OPENAI_API_KEY", "")
PINECONE_API_KEY = getenv("_PINECONE_API_KEY", "")
COHERE_API_KEY = getenv("_COHERE_API_KEY", "")
NVIDIA_API_KEY = getenv("_NVIDIA_API_KEY", "")

API_KEYS = {
    "anthropic": ANTHROPIC_API_KEY,
    "google": GOOGLE_API_KEY,
    "openai": OPENAI_API_KEY,
    "pinecone": PINECONE_API_KEY,
    "cohere": COHERE_API_KEY,
    "nvidia": NVIDIA_API_KEY    
}

PROVIDER_DEFAULTS = {
    # "provider": (provider, init_kwargs)
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
            "persist_directory": "/Users/lucasfaudman/Documents/UnifAI/scratch/gita",         
            "is_persistent": False
        },
        {}
    ),
    "pinecone": (
        "pinecone",
        {"api_key": PINECONE_API_KEY},
        {
            "serverless_spec": {"cloud": "aws", "region": "us-east-1"},
            "deletion_protection": "disabled"
            }
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
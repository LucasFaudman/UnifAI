import pytest
from os import getenv
from dotenv import load_dotenv
from unifai import ProviderConfig

load_dotenv()
ANTHROPIC_API_KEY = getenv("_ANTHROPIC_API_KEY")
GOOGLE_API_KEY = getenv("_GOOGLE_API_KEY")
OPENAI_API_KEY = getenv("_OPENAI_API_KEY")
PINECONE_API_KEY = getenv("_PINECONE_API_KEY")
COHERE_API_KEY = getenv("_COHERE_API_KEY")
NVIDIA_API_KEY = getenv("_NVIDIA_API_KEY")

api_keys = {
    "anthropic": ANTHROPIC_API_KEY,
    "google": GOOGLE_API_KEY,
    "openai": OPENAI_API_KEY,
    "pinecone": PINECONE_API_KEY,
    "cohere": COHERE_API_KEY,
    "nvidia": NVIDIA_API_KEY    
}

PROVIDER_DEFAULTS = {
    # "provider": (provider, init_kwargs, func_kwargs)

    # llms
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

    # vector dbs
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
        {
            "serverless_spec": {"cloud": "aws", "region": "us-east-1"},
            "deletion_protection": "disabled"
            }
    ),   

    # reraankers
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

    # document dbs    
    "dict": (
        "dict",
        {},
        {}
    ),
    "sqlite": (
        "sqlite",
        {},
        {}
    ),

    # document chunkers
    "count_chars_chunker": ("count_chars_chunker", {}, {}),
    "count_tokens_chunker": ("count_tokens_chunker", {}, {}),
    "count_words_chunker": ("count_words_chunker", {}, {}),

    # document loaders
    "csv_loader": ("csv_loader", {}, {}),
    "document_db_loader": ("document_db_loader", {}, {}),
    "html_loader": ("html_loader", {}, {}),
    "json_loader": ("json_loader", {}, {}),
    "markdown_loader": ("markdown_loader", {}, {}),
    "ms_office_loader": ("ms_office_loader", {}, {}),
    "pdf_loader": ("pdf_loader", {}, {}),
    "text_file_loader": ("text_file_loader", {}, {}),
    "url_loader": ("url_loader", {}, {}),

    # tokenizers
    "str_split": (
        "str_split",
        {"support_encode_decode": True},
        {}
    ),

    "tiktoken":  (
        "tiktoken",
        {},
        {}
    ),

    "huggingface":  (
        "huggingface",
        {},
        {}
    ),

    "voyage":  (
        "voyage",
        {},
        {}
    ),





}

LLM_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["anthropic"],
    PROVIDER_DEFAULTS["google"],
    PROVIDER_DEFAULTS["openai"],
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

DOCUMENT_CHUNKER_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["count_chars_chunker"], 
    PROVIDER_DEFAULTS["count_tokens_chunker"],
    PROVIDER_DEFAULTS["count_words_chunker"],
    # PROVIDER_DEFAULTS["sentence_chunker"]

]
DOCUMENT_CHUNKER_PROVIDERS = [provider[0] for provider in DOCUMENT_CHUNKER_PROVIDER_DEFAULTS]


DOCUMENT_DB_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["dict"],
    # PROVIDER_DEFAULTS["sqlite"],
    # PROVIDER_DEFAULTS["firebase"],
    # PROVIDER_DEFAULTS["mongodb"],
    # PROVIDER_DEFAULTS["elasticsearch"],
]
DOCUMENT_DB_PROVIDERS = [provider[0] for provider in DOCUMENT_DB_PROVIDER_DEFAULTS]


DOCUMENT_LOADER_PROVIDER_DEFAULTS = [
    # PROVIDER_DEFAULTS["csv_loader"],
    # PROVIDER_DEFAULTS["document_db_loader"],
    # PROVIDER_DEFAULTS["html_loader"],
    # PROVIDER_DEFAULTS["json_loader"],
    # PROVIDER_DEFAULTS["markdown_loader"],
    # PROVIDER_DEFAULTS["ms_office_loader"],
    # PROVIDER_DEFAULTS["pdf_loader"],
    PROVIDER_DEFAULTS["text_file_loader"],
    # PROVIDER_DEFAULTS["url_loader"],
]
DOCUMENT_LOADER_PROVIDERS = [provider[0] for provider in DOCUMENT_LOADER_PROVIDER_DEFAULTS]


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

TOKENIZER_PROVIDER_DEFAULTS = [
    PROVIDER_DEFAULTS["huggingface"],
    PROVIDER_DEFAULTS["str_split"],
    PROVIDER_DEFAULTS["tiktoken"],
    # PROVIDER_DEFAULTS["voyage"]
]
TOKENIZER_PROVIDERS = [provider[0] for provider in TOKENIZER_PROVIDER_DEFAULTS]


def base_test(*providers, exclude=[]):
    def decorator(func):
        return pytest.mark.parametrize(
            "provider, init_kwargs, func_kwargs", 
            [PROVIDER_DEFAULTS[provider] for provider in providers if provider not in exclude]
        )(func)
    return decorator

# parameterized test decorators


# LLM test decorators
def base_test_llms_all(func):
    return base_test(*LLM_PROVIDERS)(func)

# Document Chunker test decorators
def base_test_document_chunkers_all(func):
    return base_test(*DOCUMENT_CHUNKER_PROVIDERS)(func)

# Document DB test decorators
def base_test_document_dbs_all(func):
    return base_test(*DOCUMENT_DB_PROVIDERS)(func)

# Document Loader test decorators
def base_test_document_loaders_all(func):
    return base_test(*DOCUMENT_LOADER_PROVIDERS)(func)

# Embedding test decorators
def base_test_embeddings_all(func):
    return base_test(*EMBEDDING_PROVIDERS)(func)

# Vector DB test decorators
def base_test_vector_dbs_all(func):
    return base_test(*VECTOR_DB_PROVIDERS)(func)

# Reranker test decorators
def base_test_rerankers_all(func):
    return base_test(*RERANKER_PROVIDERS)(func)

# Tokenizer test decorators
def base_test_tokenizers_all(func):
    return base_test(*TOKENIZER_PROVIDERS)(func)







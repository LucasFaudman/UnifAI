import pytest
from os import getenv
from dotenv import load_dotenv
from unifai import ProviderConfig, DocumentChunkerConfig, DocumentDBConfig, DocumentLoaderConfig, EmbedderConfig, FunctionConfig, LLMConfig, OutputParserConfig, RAGConfig, RerankerConfig, TokenizerConfig, VectorDBConfig


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

LLM_CONFIGS = [
    LLMConfig(provider='anthropic'),
    LLMConfig(provider='google'),
    LLMConfig(provider='openai'),
    # LLMConfig(provider='ollama', init_kwargs={"host": "http://librem-2.local:11434"}),
    # LLMConfig(provider='cohere'),
    LLMConfig(provider='nvidia')    
]

DOCUMENT_CHUNKER_CONFIGS = [
    DocumentChunkerConfig(provider='count_chars_chunker'),
    DocumentChunkerConfig(provider='count_tokens_chunker'),
    DocumentChunkerConfig(provider='count_words_chunker'),
]

DOCUMENT_DB_CONFIGS = [
    DocumentDBConfig(provider='dict'),
    # DocumentDBConfig(provider='sqlite'),
    # DocumentDBConfig(provider='firebase'),
    # DocumentDBConfig(provider='mongodb'),
]

DOCUMENT_LOADER_CONFIGS = [
    # DocumentLoaderConfig(provider='csv_loader'),
    # DocumentLoaderConfig(provider='document_db_loader'),
    # DocumentLoaderConfig(provider='html_loader'),
    # DocumentLoaderConfig(provider='json_loader'),
    # DocumentLoaderConfig(provider='markdown_loader'),
    # DocumentLoaderConfig(provider='ms_office_loader'),
    # DocumentLoaderConfig(provider='pdf_loader'),
    DocumentLoaderConfig(provider='text_file_loader'),
    # DocumentLoaderConfig(provider='url_loader'),
]

EMBEDDER_CONFIGS = [
    EmbedderConfig(provider='google'),
    EmbedderConfig(provider='openai'),
    # EmbedderConfig(provider='ollama', init_kwargs={"host": "http://librem-2.local:11434"}),
    # EmbedderConfig(provider='chroma'),
    # EmbedderConfig(provider='pinecone'),
    EmbedderConfig(provider='cohere'),
    EmbedderConfig(provider='sentence_transformers'),
    EmbedderConfig(provider='nvidia')
]

FUNCTIONS_CONFIGS = [
]

OUTPUT_PARSER_CONFIGS = [
]

RAG_CONFIGS = [
]

RERANKER_CONFIGS = [
    # RerankerConfig(provider='ollama')
    RerankerConfig(provider='cohere'),
    RerankerConfig(provider='rank_bm25'),
    RerankerConfig(provider='sentence_transformers'),
    RerankerConfig(provider='nvidia')
]

TOKENIZER_CONFIGS = [
    TokenizerConfig(provider='huggingface'),
    TokenizerConfig(provider='str_split', init_kwargs={"support_encode_decode": True}),
    TokenizerConfig(provider='tiktoken'),
    # TokenizerConfig(provider='voyage')
]

VECTOR_DB_CONFIGS = [
    VectorDBConfig(provider='chroma'),
    VectorDBConfig(provider='pinecone')
]


def base_test(*configs, exclude=[]):
    def decorator(func):
        provider_init_kwargs = []
        for config in configs:                
            if isinstance(config, str) and config not in exclude:
                provider_init_kwargs.append((config, {}))        
            else:
                provider = config.provider
                init_kwargs = config.init_kwargs or {}
                if provider not in exclude:
                    provider_init_kwargs.append((provider, init_kwargs))
        
        return pytest.mark.parametrize("provider, init_kwargs", provider_init_kwargs)(func)
    return decorator

def base_test_llms(func):
    return base_test(*LLM_CONFIGS)(func)

def base_test_document_chunkers(func):
    return base_test(*DOCUMENT_CHUNKER_CONFIGS)(func)

def base_test_document_dbs(func):
    return base_test(*DOCUMENT_DB_CONFIGS)(func)

def base_test_document_loaders(func):
    return base_test(*DOCUMENT_LOADER_CONFIGS)(func)

def base_test_embedders(func):
    return base_test(*EMBEDDER_CONFIGS)(func)

def base_test_functions(func):
    return base_test(*FUNCTIONS_CONFIGS)(func)

def base_test_output_parsers(func):
    return base_test(*OUTPUT_PARSER_CONFIGS)(func)

def base_test_rag(func):
    return base_test(*RAG_CONFIGS)(func)

def base_test_rerankers(func):
    return base_test(*RERANKER_CONFIGS)(func)

def base_test_tokenizers(func):
    return base_test(*TOKENIZER_CONFIGS)(func)

def base_test_vector_dbs(func):
    return base_test(*VECTOR_DB_CONFIGS)(func)





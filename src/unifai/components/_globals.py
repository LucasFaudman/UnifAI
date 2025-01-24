COMPONENT_TYPES = [
    "chat",
    "chat_db",
    "document_chunker",
    "document_db",
    "document_loader",
    "document_transformer",
    "embedder",
    "executor",
    "function",
    "http_client",
    "llm",
    "output_parser",
    "prompt_template",
    "ragpipe",
    "reranker",
    "tokenizer",
    "toolkit",
    "tool_caller",
    "vector_db"
]

DEFAULT_PROVIDERS = {
    "chat": "default",
    "chat_db": "ephemeral",
    "document_chunker": "text_chunker",
    "document_db": "ephemeral",
    "document_loader": "default",
    "document_transformer": "default",
    "embedder": "openai",
    "executor": "default",
    "function": "default",
    "llm": "openai",
    "output_parser": "pydantic_parser",
    "prompt_template": "default",
    "ragpipe": "default",
    "reranker": "rank_bm25",
    "tokenizer": "tiktoken",
    "toolkit": "default",
    "tool_caller": "default",
    "vector_db": "chroma",
}

PROVIDERS = {
    "chat": [
        "default",
        "dynamic_memory", # TODO bfg clean repo before merge
    ],
    "chat_db": [
        "ephemeral", # TODO bfg clean repo before merge
        "firestore", # TODO bfg clean repo before merge
        "sqlite", # TODO bfg clean repo before merge
        "mongodb", # TODO bfg clean repo before merge
    ],
    "document_chunker": [
        "text_chunker", 
        "html_chunker", # TODO bfg clean repo before merge
        "json_chunker", # TODO bfg clean repo before merge
        "semantic_chunker", # TODO bfg clean repo before merge
        "unstructured" # TODO bfg clean repo before merge
    ],
    "document_db": [
        "ephemeral", 
        "firestore", # TODO bfg clean repo before merge
        "sqlite", 
        "mongodb", # TODO bfg clean repo before merge
    ],
    "document_loader": [
        "default",
        "csv_loader", # TODO bfg before merge
        "document_db_loader", # TODO bfg before merge
        "text_file_loader", 
        "json_loader", # TODO bfg before merge
        "markdown_loader", # TODO bfg before merge
        "ms_office_loader", # TODO bfg before merge
        "pdf_loader", # TODO bfg before merge
        "url_loader" # TODO bfg before merge
    ],
    "document_transformer": [
        "default",
        "metadata_updater", # TODO bfg before merge
        "reducer" # TODO bfg before merge
        "summarizer" # TODO bfg before merge
        "translator" # TODO bfg before merge
    ],
    "embedder": [
        "cohere", 
        "google", 
        "nvidia", 
        "ollama", 
        "openai", 
        "sentence_transformers"
    ],
    "function": [
        "default",
        "reasoning", # TODO bfg before merge
        "state_passing" # TODO bfg before merge
    ],
    "http_client": [ # TODO bfg clean repo before merge
        "httpx",
        "requests",
        "aiohttp",
        "selenium",
        "souperscraper",
        "playwright",
    ],
    "llm": [
        "anthropic", 
        "cohere", 
        "google", 
        "ollama", 
        "openai", 
        "nvidia"
    ],
    "output_parser": [
        "json_parser", 
        "pydantic_parser"
    ],
    "prompt_template": [
        "default",
    ],        
    "ragpipe": [
        "default", # single collection, single query
        "multi_query", # single collection, multiple queries
        "multi_collection", # multiple collections, single query
        "multi" # multiple collections, multiple queries
    ],
    "reranker": [
        "cohere", 
        "nvidia", 
        "rank_bm25", 
        "sentence_transformers",
        "voyage" # TODO bfg before merge
    ],
    "tokenizer": [
        "huggingface", 
        "str_len", 
        "str_split", 
        "tiktoken", 
        "voyage" # TODO bfg before merge
    ],
    "tool_caller": [
        "default", 
        "concurrent"
    ],
    "toolkit": [
        "default", # TODO bfg before merge
    ],
    "vector_db": [
        "chroma", 
        "pinecone"
    ],
}



from ._base_configs import ProviderConfig
from .chat_config import ChatConfig
from .document_chunker_config import DocumentChunkerConfig
from .document_db_config import DocumentDBConfig, DocumentDBCollectionConfig
from .document_loader_config import DocumentLoaderConfig
from .embedder_config import EmbedderConfig
from .function_config import FunctionConfig
from .llm_config import LLMConfig
from .output_parser_config import OutputParserConfig
from .reranker_config import RerankerConfig
from .tokenizer_config import TokenizerConfig
from .tool_caller_config import ToolCallerConfig
from .vector_db_config import VectorDBConfig, VectorDBCollectionConfig
from .rag_config import RAGConfig
from .function_config import FunctionConfig
from .unifai_config import UnifAIConfig

COMPONENT_CONFIGS = {
    "document_chunker": DocumentChunkerConfig,
    "document_db": DocumentDBConfig,
    "document_db_collection": DocumentDBCollectionConfig,
    "document_loader": DocumentLoaderConfig,
    "embedder": EmbedderConfig,
    "llm": LLMConfig,
    "output_parser": OutputParserConfig,
    "reranker": RerankerConfig,
    "tokenizer": TokenizerConfig,
    "tool_caller": ToolCallerConfig,
    "vector_db": VectorDBConfig,
    "vector_db_collection": VectorDBCollectionConfig,
    "ragpipe": RAGConfig,
    "chat": ChatConfig,
    "function": FunctionConfig,
}
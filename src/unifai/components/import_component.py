from typing import Type, Callable
from ..types.valid_inputs import ComponentType, Provider, LLMProvider, EmbeddingProvider, VectorDBProvider, RerankProvider

LLMS: frozenset[LLMProvider] = frozenset(("anthropic", "cohere", "google", "ollama", "nvidia"))
EMBEDDERS: frozenset[EmbeddingProvider] = frozenset(("cohere", "google", "nvidia", "ollama", "openai"  ,"sentence_transformers"))
VECTOR_DBS: frozenset[VectorDBProvider] = frozenset(("chroma", "pinecone"))
RERANKERS: frozenset[RerankProvider] = frozenset(("cohere", "nvidia", "rank_bm25", "sentence_transformers"))
DOCUMENT_DBS = frozenset(("dict", "sqlite", "mongo", "firebase"))
DOCUMENT_CHUNKERS = frozenset(("unstructured"))
OUTPUT_PARSERS = frozenset(("json", "pydantic"))
TOOL_CALLERS = frozenset(("default", "concurrent"))


PROVIDERS = {
    "llm": LLMS,
    "embedder": EMBEDDERS,
    "vector_db": VECTOR_DBS,
    "reranker": RERANKERS,
    "document_db": DOCUMENT_DBS,
    "document_chunker": DOCUMENT_CHUNKERS,
    "output_parser": OUTPUT_PARSERS,
    "tool_caller": TOOL_CALLERS
}    

def import_component(provider: Provider, component_type: ComponentType) -> Type|Callable:
    try:
        match component_type:
            case "llm":
                if provider == "anthropic":
                    from .llms.anhropic_llm import AnthropicLLM
                    return AnthropicLLM
                elif provider == "google":
                    from .llms.google_llm import GoogleLLM
                    return GoogleLLM
                elif provider == "nvidia":
                    from .llms.nvidia_llm import NvidiaLLM
                    return NvidiaLLM
                elif provider == "ollama":
                    from .llms.ollama_llm import OllamaLLM
                    return OllamaLLM
                elif provider == "openai":
                    from .llms.openai_llm import OpenAILLM
                    return OpenAILLM
                
            case "embedder":
                if provider == "cohere":
                    from .embedders.cohere_embedder import CohereEmbedder
                    return CohereEmbedder
                elif provider == "google":
                    from .embedders.google_embedder import GoogleEmbedder
                    return GoogleEmbedder
                elif provider == "nvidia":
                    from .embedders.nvidia_embedder import NvidiaEmbedder
                    return NvidiaEmbedder
                elif provider == "ollama":
                    from .embedders.ollama_embedder import OllamaEmbedder
                    return OllamaEmbedder
                elif provider == "openai":
                    from .embedders.openai_embedder import OpenAIEmbedder
                    return OpenAIEmbedder
                elif provider == "sentence_transformers":
                    from .embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
                    return SentenceTransformersEmbedder


            case "vector_db":
                if provider == "chroma":
                    from .retrievers.chroma_client import ChromaClient
                    return ChromaClient
                elif provider == "pinecone":
                    from .retrievers.pinecone_client import PineconeClient
                    return PineconeClient
            

            case "reranker":
                if provider == "cohere":
                    from .rerankers.cohere_reranker import CohereReranker
                    return CohereReranker
                elif provider == "nvidia":
                    from .rerankers.nvidia_reranker import NvidiaReranker
                    return NvidiaReranker
                elif provider == "rank_bm25":
                    from .rerankers.rank_bm25_reranker import RankBM25Reranker
                    return RankBM25Reranker
                elif provider == "sentence_transformers":
                    from .rerankers.sentence_transformers_reranker import SentenceTransformersReranker
                    return SentenceTransformersReranker
                
            case "document_db":
                if provider == "dict":
                    from .document_dbs.dict_doc_db import DictDocumentDB
                    return DictDocumentDB
                elif provider == "sqlite":
                    from .document_dbs.sqlite_doc_db import SQLiteDocumentDB
                    return SQLiteDocumentDB
                elif provider == "mongo":
                    raise NotImplementedError("MongoDB DocumentDB not yet implemented")
                elif provider == "firebase":
                    raise NotImplementedError("Firebase DocumentDB not yet implemented")
                
            case "document_chunker":
                if provider == "unstructured":
                    raise NotImplementedError("Unstructured Document Chunker not yet implemented")
            
            case "output_parser":
                if provider == "json":
                    from .output_parsers.json_output_parser import json_parse
                    return json_parse
                elif provider == "pydantic":
                    from .output_parsers.pydantic_output_parser import pydantic_parse
                    return pydantic_parse
            
            case "tool_caller":
                if provider == "default":
                    from .tool_callers._base_tool_caller import ToolCaller
                    return ToolCaller
                elif provider == "concurrent":
                    from .tool_callers.concurrent_tool_caller import ConcurrentToolCaller
                    return ConcurrentToolCaller
            
            case _:
                raise ValueError(f"Invalid component_type: {component_type}. Must be one of: 'llm', 'embedder', 'vector_db', 'reranker', 'document_db', 'document_chunker', 'output_parser', 'tool_caller'")
        raise ValueError(f"Invalid {component_type} provider: {provider}. Must be one of: {PROVIDERS[component_type]}")
    except ImportError as e:
        raise ValueError(f"Could not import {component_type} for {provider}. Error: {e}")
from typing import Type, Callable
from ..types.valid_inputs import ComponentType, Provider, LLMProvider, EmbeddingProvider, VectorDBProvider, RerankProvider

LLMS: frozenset[LLMProvider] = frozenset(("anthropic", "cohere", "google", "ollama", "nvidia"))
EMBEDDERS: frozenset[EmbeddingProvider] = frozenset(("cohere", "google", "nvidia", "ollama", "openai"  ,"sentence_transformers"))
VECTOR_DBS: frozenset[VectorDBProvider] = frozenset(("chroma", "pinecone"))
RERANKERS: frozenset[RerankProvider] = frozenset(("cohere", "nvidia", "rank_bm25", "sentence_transformers"))
DOCUMENT_DBS = frozenset(("dict", "sqlite", "mongo", "firebase"))


PROVIDERS = {
    "llm": LLMS,
    "embedder": EMBEDDERS,
    "vector_db": VECTOR_DBS,
    "reranker": RERANKERS,
    "document_db": DOCUMENT_DBS
}    

def import_component(provider: Provider, component_type: ComponentType) -> Type|Callable:
    try:
        match component_type:
            case "llm":
                if provider == "anthropic":
                    from unifai.components.llms.anhropic_llm import AnthropicLLM
                    return AnthropicLLM
                elif provider == "google":
                    from unifai.components.llms.google_llm import GoogleLLM
                    return GoogleLLM
                elif provider == "nvidia":
                    from unifai.components.llms.nvidia_llm import NvidiaLLM
                    return NvidiaLLM
                elif provider == "ollama":
                    from unifai.components.llms.ollama_llm import OllamaLLM
                    return OllamaLLM
                elif provider == "openai":
                    from unifai.components.llms.openai_llm import OpenAILLM
                    return OpenAILLM
                
            case "embedder":
                if provider == "cohere":
                    from unifai.components.embedders.cohere_embedder import CohereEmbedder
                    return CohereEmbedder
                elif provider == "google":
                    from unifai.components.embedders.google_embedder import GoogleEmbedder
                    return GoogleEmbedder
                elif provider == "nvidia":
                    from unifai.components.embedders.nvidia_embedder import NvidiaEmbedder
                    return NvidiaEmbedder
                elif provider == "ollama":
                    from unifai.components.embedders.ollama_embedder import OllamaEmbedder
                    return OllamaEmbedder
                elif provider == "openai":
                    from unifai.components.embedders.openai_embedder import OpenAIEmbedder
                    return OpenAIEmbedder
                elif provider == "sentence_transformers":
                    from unifai.components.embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
                    return SentenceTransformersEmbedder


            case "vector_db":
                if provider == "chroma":
                    from unifai.components.retreivers.chroma_client import ChromaClient
                    return ChromaClient
                elif provider == "pinecone":
                    from unifai.components.retreivers.pinecone_client import PineconeClient
                    return PineconeClient
            

            case "reranker":
                if provider == "cohere":
                    from unifai.components.rerankers.cohere_reranker import CohereReranker
                    return CohereReranker
                elif provider == "nvidia":
                    from unifai.components.rerankers.nvidia_reranker import NvidiaReranker
                    return NvidiaReranker
                elif provider == "rank_bm25":
                    from unifai.components.rerankers.rank_bm25_reranker import RankBM25Reranker
                    return RankBM25Reranker
                elif provider == "sentence_transformers":
                    from unifai.components.rerankers.sentence_transformers_reranker import SentenceTransformersReranker
                    return SentenceTransformersReranker
                
            case "document_db":
                if provider == "dict":
                    from unifai.components.document_dbs.dict_doc_db import DictDocumentDB
                    return DictDocumentDB
                elif provider == "sqlite":
                    from unifai.components.document_dbs.sqlite_doc_db import SQLiteDocumentDB
                    return SQLiteDocumentDB
                elif provider == "mongo":
                    raise NotImplementedError("MongoDB DocumentDB not yet implemented")
                elif provider == "firebase":
                    raise NotImplementedError("Firebase DocumentDB not yet implemented")
                
            case "document_chunker":
                pass
            case "output_parser":
                pass
            case "tool_caller":
                pass
            case _:
                raise ValueError(f"Invalid component_type: {component_type}. Must be one of: 'llm', 'embedder', 'vector_db', 'reranker', 'document_db', 'document_chunker', 'output_parser', 'tool_caller'")
        raise ValueError(f"Invalid {component_type} provider: {provider}. Must be one of: {PROVIDERS[component_type]}")
    except ImportError as e:
        raise ValueError(f"Could not import {component_type} for {provider}. Error: {e}")
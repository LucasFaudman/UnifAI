from typing import Type, Callable
from ..types.annotations import ComponentType, ProviderName

PROVIDERS = {
    "llm": frozenset(("anthropic", "cohere", "google", "ollama", "nvidia")),
    "embedder": frozenset(("cohere", "google", "nvidia", "ollama", "openai"  ,"sentence_transformers")),
    "vector_db": frozenset(("chroma", "pinecone")),
    "reranker": frozenset(("cohere", "nvidia", "rank_bm25", "sentence_transformers")),
    "document_db": frozenset(("dict", "sqlite", "mongo", "firebase")),
    "document_chunker": frozenset(("count_chars_chunker", "count_tokens_chunker", "count_words_chunker", "code_chunker", "html_chunker", "json_chunker", "semantic_chunker", "unstructured")),
    "document_loader": frozenset(("csv_loader", "document_db_loader", "html_loader", "json_loader", "markdown_loader", "ms_office_loader", "pdf_loader", "text_file_loader", "url_loader")),
    "output_parser": frozenset(("json_parser", "pydantic_parser")),
    "ragpipe": frozenset(("default", "concurrent")),
    "tool_caller": frozenset(("default", "concurrent")),
    "tokenizer": frozenset(("huggingface", "str_len", "str_split", "tiktoken", "voyage"))
}    

def import_component(component_type: ComponentType, provider: ProviderName) -> Type|Callable:
    try:
        match component_type:
            case "llm":
                if provider == "anthropic":
                    from .llms.anhropic_llm import AnthropicLLM
                    return AnthropicLLM
                if provider == "google":
                    from .llms.google_llm import GoogleLLM
                    return GoogleLLM
                if provider == "nvidia":
                    from .llms.nvidia_llm import NvidiaLLM
                    return NvidiaLLM
                if provider == "ollama":
                    from .llms.ollama_llm import OllamaLLM
                    return OllamaLLM
                if provider == "openai":
                    from .llms.openai_llm import OpenAILLM
                    return OpenAILLM
                
            case "embedder":
                if provider == "cohere":
                    from .embedders.cohere_embedder import CohereEmbedder
                    return CohereEmbedder
                if provider == "google":
                    from .embedders.google_embedder import GoogleEmbedder
                    return GoogleEmbedder
                if provider == "nvidia":
                    from .embedders.nvidia_embedder import NvidiaEmbedder
                    return NvidiaEmbedder
                if provider == "ollama":
                    from .embedders.ollama_embedder import OllamaEmbedder
                    return OllamaEmbedder
                if provider == "openai":
                    from .embedders.openai_embedder import OpenAIEmbedder
                    return OpenAIEmbedder
                if provider == "sentence_transformers":
                    from .embedders.sentence_transformers_embedder import SentenceTransformersEmbedder
                    return SentenceTransformersEmbedder


            case "vector_db":
                if provider == "chroma":
                    from .vector_dbs.chroma_vector_db import ChromaVectorDB
                    return ChromaVectorDB
                if provider == "pinecone":
                    from .vector_dbs.pinecone_vector_db import PineconeVectorDB
                    return PineconeVectorDB
            
            case "reranker":
                if provider == "cohere":
                    from .rerankers.cohere_reranker import CohereReranker
                    return CohereReranker
                if provider == "nvidia":
                    from .rerankers.nvidia_reranker import NvidiaReranker
                    return NvidiaReranker
                if provider == "rank_bm25":
                    from .rerankers.rank_bm25_reranker import RankBM25Reranker
                    return RankBM25Reranker
                if provider == "sentence_transformers":
                    from .rerankers.sentence_transformers_reranker import SentenceTransformersReranker
                    return SentenceTransformersReranker
                
            case "document_db":
                if provider == "dict":
                    from .document_dbs.dict_doc_db import DictDocumentDB
                    return DictDocumentDB
                if provider == "sqlite":
                    from .document_dbs.sqlite_doc_db import SQLiteDocumentDB
                    return SQLiteDocumentDB
                if provider == "mongo":
                    raise NotImplementedError("MongoDB DocumentDB not yet implemented")
                if provider == "firebase":
                    raise NotImplementedError("Firebase DocumentDB not yet implemented")
                
            case "document_chunker":
                if provider == "count_chars_chunker":
                    from .document_chunkers.count_chars_chunker import CountCharsDocumentChunker
                    return CountCharsDocumentChunker
                if provider == "count_tokens_chunker":
                    from .document_chunkers.count_tokens_chunker import CountTokensDocumentChunker
                    return CountTokensDocumentChunker
                if provider == "count_words_chunker":
                    from .document_chunkers.count_words_chunker import CountWordsDocumentChunker
                    return CountWordsDocumentChunker
                if provider == "code_chunker":
                    from .document_chunkers.code_chunker import CodeDocumentChunker
                    return CodeDocumentChunker
                if provider == "html_chunker":
                    from .document_chunkers.html_chunker import HTMLDocumentChunker
                    return HTMLDocumentChunker
                if provider == "json_chunker":
                    from .document_chunkers.json_chunker import JSONDocumentChunker
                    return JSONDocumentChunker
                if provider == "semantic_chunker":
                    from .document_chunkers.semantic_chunker import SemanticDocumentChunker
                    return SemanticDocumentChunker
                if provider == "unstructured":
                    raise NotImplementedError("Unstructured Document Chunker not yet implemented")
            
            case "document_loader":
                if provider == "document_db_loader":
                    from .document_loaders.document_db_loader import DocumentDBLoader
                    return DocumentDBLoader
                if provider == "html_loader":
                    from .document_loaders.html_loader import HTMLDocumentLoader
                    return HTMLDocumentLoader
                if provider == "json_loader":
                    from .document_loaders.json_loader import JSONDocumentLoader
                    return JSONDocumentLoader
                if provider == "markdown_loader":
                    from .document_loaders.markdown_loader import MarkdownDocumentLoader
                    return MarkdownDocumentLoader
                if provider == "ms_office_loader":
                    from .document_loaders.ms_office_loader import MSOfficeDocumentLoader
                    return MSOfficeDocumentLoader
                if provider == "pdf_loader":
                    from .document_loaders.pdf_loader import PDFDocumentLoader
                    return PDFDocumentLoader
                if provider == "text_file_loader":
                    from .document_loaders.text_file_loader import TextFileDocumentLoader
                    return TextFileDocumentLoader
                if provider == "url_loader":
                    from .document_loaders.url_loader import URLDocumentLoader
                    return URLDocumentLoader


            case "output_parser":
                if provider == "json_parser":
                    from .output_parsers.json_output_parser import JSONParser
                    return JSONParser
                if provider == "pydantic_parser":
                    from .output_parsers.pydantic_output_parser import PydanticParser
                    return PydanticParser
            
            case "ragpipe":
                if provider == "default":
                    from .ragpipe import RAGPipe
                    return RAGPipe

            case "tool_caller":
                if provider == "default":
                    from ._base_components._base_tool_caller import ToolCaller
                    return ToolCaller
                if provider == "concurrent":
                    from .tool_callers.concurrent_tool_caller import ConcurrentToolCaller
                    return ConcurrentToolCaller
                
            case "tokenizer":
                if provider == "huggingface":
                    from .tokenizers.huggingface_tokenizer import HuggingFaceTokenizer
                    return HuggingFaceTokenizer
                if provider == "str_len":
                    from .tokenizers.str_test_tokenizers import StrLenTokenizer
                    return StrLenTokenizer  
                if provider == "str_split":
                    from .tokenizers.str_test_tokenizers import StrSplitTokenizer
                    return StrSplitTokenizer    
                if provider == "tiktoken":
                    from .tokenizers.tiktoken_tokenizer import TikTokenTokenizer
                    return TikTokenTokenizer
                if provider == "voyage":
                    from .tokenizers.voyage_tokenizer import VoyageTokenizer
                    return VoyageTokenizer            
            case _:
                raise ValueError(f"Invalid component_type: {component_type}. Must be one of: {list(PROVIDERS.keys())}")
        raise ValueError(f"Invalid {component_type} provider: {provider}. Must be one of: {PROVIDERS[component_type]}")
    except ImportError as e:
        raise ValueError(f"Could not import {component_type} for {provider}. Error: {e}")
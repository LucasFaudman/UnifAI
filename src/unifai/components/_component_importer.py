from typing import TYPE_CHECKING, Type, Callable, Any
if TYPE_CHECKING:
    from ..types.annotations import ComponentType, ProviderName

from importlib import import_module

class ComponentImporter:
    """
    Lazy import mechanism for UnifAI components.
    """

    _COMPONENT_PATHS = {
        "chat": {
            "default": ".chats.default_chat.Chat",
        },
        "chat_db": {
        },
        "document_chunker": {
            "html_chunker": ".document_chunkers.html_chunker.HTMLDocumentChunker",
            "json_chunker": ".document_chunkers.json_chunker.JSONDocumentChunker",
            "semantic_chunker": ".document_chunkers.semantic_chunker.SemanticDocumentChunker",
            "text_chunker": ".document_chunkers.text_chunker.TextDocumentChunker",
        },
        "document_db": {
            "ephemeral": ".document_dbs.ephemeral_document_db.EphemeralDocumentDB",
            "firestore": ".document_dbs.firestore_docu_db.FirestoreDocumentDB",
            "sqlite": ".document_dbs.sqlite_document_db.SQLiteDocumentDB",
        },
        "document_loader": {
            "default": ".document_loaders.default_loader.DocumentLoader",
            "document_db_loader": ".document_loaders.document_db_loader.DocumentDBLoader",
            "html_loader": ".document_loaders.html_loader.HTMLDocumentLoader",
            "json_loader": ".document_loaders.json_loader.JSONDocumentLoader",
            "markdown_loader": ".document_loaders.markdown_loader.MarkdownDocumentLoader",
            "ms_office_loader": ".document_loaders.ms_office_loader.MSOfficeDocumentLoader",
            "pdf_loader": ".document_loaders.pdf_loader.PDFDocumentLoader",
            "text_file_loader": ".document_loaders.text_file_loader.TextFileDocumentLoader",
            "url_loader": ".document_loaders.url_loader.URLDocumentLoader",
        },
        "document_transformer": {
        },
        "embedder": {
            "cohere": ".embedders.cohere_embedder.CohereEmbedder",
            "google": ".embedders.google_embedder.GoogleEmbedder",
            "nvidia": ".embedders.nvidia_embedder.NvidiaEmbedder",
            "ollama": ".embedders.ollama_embedder.OllamaEmbedder",
            "openai": ".embedders.openai_embedder.OpenAIEmbedder",
            "sentence_transformers": ".embedders.sentence_transformers_embedder.SentenceTransformersEmbedder",
        },
        "function": {
            "default": ".functions.default_function.Function",
        },
        "llm": {
            "anthropic": ".llms.anhropic_llm.AnthropicLLM",
            "deepseek": ".llms.deepseek_llm.DeepSeekLLM",
            "google": ".llms.google_llm.GoogleLLM",
            "nvidia": ".llms.nvidia_llm.NvidiaLLM",
            "ollama": ".llms.ollama_llm.OllamaLLM",
            "openai": ".llms.openai_llm.OpenAILLM",
        },
        "output_parser": {
            "json_parser": ".output_parsers.json_output_parser.JSONParser",
            "pydantic_parser": ".output_parsers.pydantic_output_parser.PydanticParser",
        },
        "ragpipe": {
            "default": ".ragpipes.default_ragpipe.RAGPipe",
        },
        "reranker": {
            "cohere": ".rerankers.cohere_reranker.CohereReranker",
            "nvidia": ".rerankers.nvidia_reranker.NvidiaReranker",
            "rank_bm25": ".rerankers.rank_bm25_reranker.RankBM25Reranker",
            "sentence_transformers": ".rerankers.sentence_transformers_reranker.SentenceTransformersReranker",
        },
        "tool_caller": {
            "default": ".tool_callers.ToolCaller",
            "concurrent": ".tool_callers.concurrent_tool_caller.ConcurrentToolCaller",
        },
        "tokenizer": {
            "huggingface": ".tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
            "str_len": ".tokenizers.str_test_tokenizers.StrLenTokenizer",
            "str_split": ".tokenizers.str_test_tokenizers.StrSplitTokenizer",
            "tiktoken": ".tokenizers.tiktoken_tokenizer.TikTokenTokenizer",
            "voyage": ".tokenizers.voyage_tokenizer.VoyageTokenizer",
        },
        "vector_db": {
            "chroma": ".vector_dbs.chroma_vector_db.ChromaVectorDB",
            "pinecone": ".vector_dbs.pinecone_vector_db.PineconeVectorDB",
        }
    }

    @classmethod
    def import_component(cls, component_type: "ComponentType", provider: "ProviderName") -> Type[Any] | Callable[..., Any]:
        """
        Lazily import and return a specific component based on type and provider.
        
        Args:
            component_type (str): The type of component to import
            provider (str): The specific provider/implementation to use
        
        Returns:
            The imported component class or factory function
        
        Raises:
            ImportError: If the component type or provider is invalid
        """
        # Validate component type
        if component_type not in cls._COMPONENT_PATHS:
            raise ImportError(f"Invalid component type: {component_type}. " 
                              f"Available types: {list(cls._COMPONENT_PATHS.keys())}")
        
        # Validate provider
        providers = cls._COMPONENT_PATHS[component_type]
        if provider not in providers:
            raise ImportError(f"Invalid provider '{provider}' for component type '{component_type}'. " 
                              f"Available providers: {list(providers.keys())}")
        
        # Lazily import the module
        try:
            import_path = providers[provider]
            package = __package__ if import_path.startswith('.') else None
            module_path, class_name = import_path.rsplit('.', 1)            
            module = import_module(module_path, package)
            # Return the specific class or function
            return getattr(module, class_name)
        
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import component: {component_type} "
                f"with provider: {provider}. Error: {e}"
            ) from e
        
    @classmethod
    def register_for_import(cls, component_type: "ComponentType", provider: "ProviderName", import_path: str) -> None:
        """
        Register a custom import path for a specific component type and provider.
        
        Args:
            component_type (str): The type of component to register
            provider (str): The specific provider/implementation to register
            import_path (str): The import path for the component
        """
        if component_type not in cls._COMPONENT_PATHS:
            cls._COMPONENT_PATHS[component_type] = {}
        if provider in cls._COMPONENT_PATHS[component_type]:
            raise ValueError(f"Import path for {component_type} and {provider} already exists")
        
        cls._COMPONENT_PATHS[component_type][provider] = import_path


def import_component(component_type: "ComponentType", provider: "ProviderName") -> Type[Any] | Callable[..., Any]:
    """
    Convenience wrapper for LazyImporter.import_component
    
    Args:
        component_type (str): The type of component to import
        provider (str): The specific provider/implementation to use
    """
    return ComponentImporter.import_component(component_type, provider)

def register_for_import(component_type: "ComponentType", provider: "ProviderName", import_path: str) -> None:
    """
    Convenience wrapper for LazyImporter.register_for_import

    Args:
        component_type (str): The type of component to register
        provider (str): The specific provider/implementation to register
        import_path (str): The import path for the component
    """
    return ComponentImporter.register_for_import(component_type, provider, import_path)
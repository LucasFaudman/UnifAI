from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator
from typing import overload

from unifai.components.llms._base_llm_client import LLMClient
from unifai.components.embedders._base_embedder import Embedder
from unifai.components.rerankers._base_reranker import Reranker
from unifai.components.document_dbs._base_document_db import DocumentDB

from unifai.components.retreivers._base_vector_db_client import (
    VectorDBClient, 
    VectorDBIndex,
    VectorDBGetResult,
    VectorDBQueryResult
)

from unifai.types import (
    ComponentType,
    LLMProvider,
    EmbeddingProvider,
    VectorDBProvider,
    RerankProvider,
    Provider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    Embeddings,
    Embedding,
    VectorDBQueryResult    
)

from unifai.type_conversions import standardize_tools, standardize_specs

from ..components.tool_callers import ToolCaller
from .chat import Chat
from .rag_engine import RAGEngine
from .ai_func import AIFunction
from .specs import RAGSpec, FuncSpec


LLMS: frozenset[LLMProvider] = frozenset(("anthropic", "cohere", "google", "nvidia", ))
EMBEDDERS: frozenset[EmbeddingProvider] = frozenset(("cohere", "google", "nvidia", "ollama", "openai"  ,"sentence_transformers"))
VECTOR_DBS: frozenset[VectorDBProvider] = frozenset(("chroma", "pinecone"))
RERANKERS: frozenset[RerankProvider] = frozenset(("cohere", "nvidia", "rank_bm25", "sentence_transformers"))
REQUIRED_BOUND_METHODS: dict[Provider, list[str]] = {
    "chroma": ["embed"],
    "pinecone": ["embed"],
}
DOCUMENT_DBS = frozenset(("dict", "sqlite", "mongo", "firebase"))


PROVIDERS = {
    "llm": LLMS,
    "embedder": EMBEDDERS,
    "vector_db": VECTOR_DBS,
    "reranker": RERANKERS,
    "document_db": DOCUMENT_DBS
    
}


class UnifAIClient:
    FUNC_SPECS: list[FuncSpec|dict] = [] #  | dict[str, EvalSpec|dict] = []
    RAG_SPECS: list[RAGSpec|dict] = [] # | dict[str, EvalSpec|dict] = []
    TOOLS: list[Tool|dict] = [] # |dict[str, Tool|dict] = []
    TOOL_CALLABLES: dict[str, Callable] = {}
    
    def __init__(
            self, 
            provider_client_kwargs: Optional[dict[Provider, dict[str, Any]]] = None,
            tools: Optional[list[ToolInput]] = None,
            tool_callables: Optional[dict[str, Callable]] = None,
            func_specs: Optional[list[FuncSpec|dict]] = None,
            rag_specs: Optional[list[RAGSpec]] = None,
            default_llm_provider: Optional[LLMProvider] = None,
            default_embedding_provider: Optional[EmbeddingProvider] = None,
            default_vector_db_provider: Optional[VectorDBProvider] = None, 
            default_rerank_provider: Optional[RerankProvider] = None                
    ) -> None:
        
        self.set_provider_client_kwargs(provider_client_kwargs)
        self.set_default_llm_provider(default_llm_provider)
        self.set_default_embedding_provider(default_embedding_provider)
        self.set_default_vector_db_provider(default_vector_db_provider)
        self.set_default_rerank_provider(default_rerank_provider)
        
        self._components: dict[ComponentType, dict[Provider, LLMClient|Embedder|VectorDBClient|Reranker]] = {}
        self.tools: dict[str, Tool] = {}
        self.tool_callables: dict[str, Callable] = {}
        self.func_specs: dict[str, FuncSpec] = {}
        self.rag_specs: dict[str, RAGSpec] = {}
        
        self.add_tools(tools or self.TOOLS)
        self.add_tool_callables(tool_callables or self.TOOL_CALLABLES)
        self.add_func_specs(func_specs or self.FUNC_SPECS)
        self.add_rag_specs(rag_specs or self.RAG_SPECS)
        

    def set_provider_client_kwargs(self, provider_client_kwargs: Optional[dict[Provider, dict[str, Any]]] = None):
        self.provider_client_kwargs = provider_client_kwargs if provider_client_kwargs is not None else {}        
        self.providers: list[Provider] = list(self.provider_client_kwargs.keys())
        self.llm_providers: list[LLMProvider] = [provider for provider in self.providers if provider in LLMS]
        self.embedding_providers: list[EmbeddingProvider] = [provider for provider in self.providers if provider in EMBEDDERS]
        self.vector_db_providers: list[VectorDBProvider] = [provider for provider in self.providers if provider in VECTOR_DBS]
        self.rerank_providers: list[RerankProvider] = [provider for provider in self.providers if provider in RERANKERS]        


    def set_default_llm_provider(self, provider: Optional[LLMProvider] = None, check: bool = True):
        if check and provider and provider not in LLMS:
            raise ValueError(f"Invalid LLM provider: {provider}. Must be one of: {LLMS}")
        if provider:
            self.default_llm_provider: LLMProvider = provider
        elif self.llm_providers:
            self.default_llm_provider = self.llm_providers[0]
        else:
            self.default_llm_provider = "openai"


    def set_default_embedding_provider(self, provider: Optional[EmbeddingProvider] = None, check: bool = True):
        if check and provider and provider not in EMBEDDERS:
            raise ValueError(f"Invalid Embedding provider: {provider}. Must be one of: {EMBEDDERS}")
        if provider:
            self.default_embedding_provider: EmbeddingProvider = provider
        elif self.embedding_providers:
            self.default_embedding_provider = self.embedding_providers[0]
        else:
            self.default_embedding_provider = "openai"


    def set_default_vector_db_provider(self, provider: Optional[VectorDBProvider] = None, check: bool = True):
        if check and provider and provider not in VECTOR_DBS:
            raise ValueError(f"Invalid Vector DB provider: {provider}. Must be one of: {VECTOR_DBS}")
        if provider:
            self.default_vector_db_provider: VectorDBProvider = provider
        elif self.vector_db_providers:
            self.default_vector_db_provider = self.vector_db_providers[0]
        else:
            self.default_vector_db_provider = "chroma"


    def set_default_rerank_provider(self, provider: Optional[RerankProvider] = None, check: bool = True):
        if check and provider and provider not in RERANKERS:
            raise ValueError(f"Invalid Vector DB provider: {provider}. Must be one of: {RERANKERS}")
        if provider:
            self.default_rerank_provider: RerankProvider = provider
        elif self.rerank_providers:
            self.default_rerank_provider = self.rerank_providers[0]
        else:
            self.default_rerank_provider = "cohere"            


    def add_tools(self, tools: Optional[list[ToolInput]]):
        if not tools: return

        for tool_name, tool in standardize_tools(tools).items():
            self.tools[tool_name] = tool
            if (tool_callable := getattr(tool, "callable", None)) is not None:
                self.tool_callables[tool_name] = tool_callable

    
    def add_tool_callables(self, tool_callables: Optional[dict[str, Callable]]):
        if not tool_callables: return
        self.tool_callables.update(tool_callables)


    def add_func_specs(self, func_specs: Optional[list[FuncSpec|dict]]):
        if func_specs:
            self.func_specs.update(standardize_specs(func_specs, FuncSpec))


    def add_rag_specs(self, rag_specs: Optional[list[RAGSpec|dict]]):
        if rag_specs:
            self.rag_specs.update(standardize_specs(rag_specs, RAGSpec))




    def import_component(self, provider: Provider, component_type: ComponentType) -> Type|Callable:
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
    
    def init_component(self, provider: Provider, component_type: ComponentType, **client_kwargs) -> LLMClient|Embedder|VectorDBClient|Reranker:
        if (registered_kwargs := self.provider_client_kwargs.get(provider)) is not None:
            client_kwargs = {**registered_kwargs, **client_kwargs}
        else:
            self.provider_client_kwargs[provider] = client_kwargs
            client_kwargs = {**self.provider_client_kwargs[provider], **client_kwargs}
           
        if required_bound_methods := REQUIRED_BOUND_METHODS.get(provider):
            for method in required_bound_methods:
                if method not in client_kwargs:
                    client_kwargs[method] = getattr(self, method)
        if component_type not in self._components:
            self._components[component_type] = {}
        self._components[component_type][provider] = self.import_component(provider, component_type)(**client_kwargs)
        return self._components[component_type][provider]  


    @overload
    def get_component(self, provider: LLMProvider, component_type: ComponentType = "llm", **client_kwargs) -> LLMClient:
        ...

    @overload
    def get_component(self, provider: EmbeddingProvider, component_type: ComponentType = "embedder", **client_kwargs) -> Embedder:
        ...        

    @overload
    def get_component(self, provider: VectorDBProvider, component_type: ComponentType = "document_db", **client_kwargs) -> VectorDBClient:
        ...        

    @overload
    def get_component(self, provider: RerankProvider, component_type: ComponentType = "reranker", **client_kwargs) -> Reranker:
        ...

    def get_component(self, provider: Provider, component_type: ComponentType = "llm", **client_kwargs) -> LLMClient|Embedder|VectorDBClient|Reranker:
        components_of_type = self._components.get(component_type)
        if (not components_of_type
            or provider not in components_of_type 
            or (client_kwargs and components_of_type[provider].client_kwargs != client_kwargs
            )):
            return self.init_component(provider, component_type, **client_kwargs)
        return self._components[component_type][provider]


    def get_llm_client(self, provider: Optional[LLMProvider] = None, **client_kwargs) -> LLMClient:
        provider = provider or self.default_llm_provider
        return self.get_component(provider, **client_kwargs)


    def get_embedder(self, provider: Optional[EmbeddingProvider] = None, **client_kwargs) -> Embedder:
        provider = provider or self.default_embedding_provider
        return self.get_component(provider, **client_kwargs)


    def get_reranker(self, provider: Optional[RerankProvider] = None, **client_kwargs) -> Reranker:
        provider = provider or self.default_rerank_provider
        return self.get_component(provider, **client_kwargs)


    def get_vector_db(self, provider: Optional[VectorDBProvider] = None, **client_kwargs) -> VectorDBClient:
        provider = provider or self.default_vector_db_provider
        return self.get_component(provider, **client_kwargs)


    def get_default_model(self, provider: Provider, model_type: Literal["llm", "embedding", "rerank"]) -> str:        
        if model_type == "llm":
            return self.get_llm_client(provider).default_model
        elif model_type == "embedding":
            return self.get_embedder(provider).default_embedding_model
        elif model_type == "rerank":
            return self.get_reranker(provider).default_reranking_model
        else:
            return ValueError(f"Invalid model_type: {model_type}. Must be one of: 'llm', 'embedding', 'rerank'")



    # List Models
    def list_models(self, provider: Optional[LLMProvider] = None) -> list[str]:
        return self.get_component(provider, "llm").list_models()
    

    # Chat
    def start_chat(
            self,
            messages: Optional[Sequence[MessageInput]] = None,
            provider: Optional[LLMProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,
            return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            response_format: Optional[Union[str, dict[str, str]]] = None,
            tools: Optional[Sequence[ToolInput]] = None,            
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            enforce_tool_choice: bool = True,
            tool_choice_error_retries: int = 3,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller,
            tool_caller_kwargs: Optional[dict[str, Any]] = None,
            max_messages_per_run: int = 10,
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,

    ) -> Chat:
        provider = provider or self.default_llm_provider
        if tool_caller_class_or_instance is not None:
            tool_caller = self.get_tool_caller(tool_caller_class_or_instance, tools, tool_callables, tool_caller_kwargs)
        else:
            tool_caller = None               
        return Chat(
            get_client=self.get_llm_client,
            parent_tools=self.tools,
            messages=messages,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            return_on=return_on,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            tool_caller=tool_caller,                
            enforce_tool_choice=enforce_tool_choice,
            tool_choice_error_retries=tool_choice_error_retries,
            max_messages_per_run=max_messages_per_run,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    
    def chat(
            self,
            messages: Optional[Sequence[MessageInput]] = None,
            provider: Optional[LLMProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,             
            return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            response_format: Optional[Union[str, dict[str, str]]] = None,
            tools: Optional[Sequence[ToolInput]] = None,            
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            enforce_tool_choice: bool = True,
            tool_choice_error_retries: int = 3,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller,
            tool_caller_kwargs: Optional[dict[str, Any]] = None,
            max_messages_per_run: int = 10,
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            **kwargs
    ) -> Chat:
        chat = self.start_chat(
            messages=messages,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            return_on=return_on,
            response_format=response_format,            
            tools=tools,
            tool_choice=tool_choice,
            enforce_tool_choice=enforce_tool_choice,
            tool_choice_error_retries=tool_choice_error_retries,
            tool_callables=tool_callables,
            tool_caller_class_or_instance=tool_caller_class_or_instance,
            tool_caller_kwargs=tool_caller_kwargs,
            max_messages_per_run=max_messages_per_run,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if messages:
            chat.run(**kwargs)
        return chat
        

    def chat_stream(
            self,
            messages: Optional[Sequence[MessageInput]] = None,
            provider: Optional[LLMProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,             
            return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            response_format: Optional[Union[str, dict[str, str]]] = None,
            tools: Optional[Sequence[ToolInput]] = None,            
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            enforce_tool_choice: bool = True,
            tool_choice_error_retries: int = 3,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller,
            tool_caller_kwargs: Optional[dict[str, Any]] = None,
            max_messages_per_run: int = 10,
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            **kwargs
    ) -> Generator[MessageChunk, None, Chat]:
        chat = self.start_chat(
            messages=messages,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            return_on=return_on,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            enforce_tool_choice=enforce_tool_choice,
            tool_choice_error_retries=tool_choice_error_retries,
            tool_callables=tool_callables,
            tool_caller_class_or_instance=tool_caller_class_or_instance,
            tool_caller_kwargs=tool_caller_kwargs,
            max_messages_per_run=max_messages_per_run,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if messages:
            yield from chat.run_stream(**kwargs)
        return chat
    

    def embed(
        self, 
        input: str | Sequence[str],
        model: Optional[str] = None,
        provider: Optional[EmbeddingProvider] = None,
        dimensions: Optional[int] = None,
        task_type: Optional[Literal[
        "retreival_query", 
        "retreival_document", 
        "semantic_similarity", 
        "classification", 
        "clustering", 
        "question_answering", 
        "fact_verification", 
        "code_retreival_query", 
        "image"]] = None,
        input_too_large: Literal[
        "truncate_end", 
        "truncate_start", 
        "raise_error"] = "truncate_end",
        dimensions_too_large: Literal[
        "reduce_dimensions", 
        "raise_error"
        ] = "reduce_dimensions",
        task_type_not_supported: Literal[
        "use_closest_supported",
        "raise_error",
        ] = "use_closest_supported",                 
        **kwargs
              ) -> Embeddings:
        
        return self.get_embedder(provider).embed(
            input, model, dimensions, task_type, input_too_large, dimensions_too_large, task_type_not_supported, **kwargs)


    

    def get_or_create_index(self, 
                            name: str,
                            vector_db_provider: Optional[VectorDBProvider] = None,                            
                            embedding_provider: Optional[EmbeddingProvider] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                            index_metadata: Optional[dict] = None,
                            **kwargs
                            ) -> VectorDBIndex:
        return self.get_vector_db(vector_db_provider).get_or_create_index(
            name=name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            metadata=index_metadata,
            **kwargs
        )        
        # return self.get_vector_db_client(vector_db_provider).get_or_create_index(
        #     name=name,
        #     embedding_provider=embedding_provider,
        #     embedding_model=embedding_model,
        #     dimensions=dimensions,
        #     distance_metric=distance_metric,
        #     metadata=index_metadata,
        #     **kwargs
        # )
    

    def upsert_index(self,
                     name: str,
                     ids: list[str],
                     metadatas: Optional[list[dict]] = None,
                     documents: Optional[list[str]] = None,
                     embeddings: Optional[list[Embedding]] = None,
                     vector_db_provider: Optional[VectorDBProvider] = None,                            
                     embedding_provider: Optional[LLMProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                     index_metadata: Optional[dict] = None
                     ) -> VectorDBIndex:
        return self.get_or_create_index(
              name=name,
              vector_db_provider=vector_db_provider,
              embedding_provider=embedding_provider,
              embedding_model=embedding_model,
              dimensions=dimensions,
              distance_metric=distance_metric,
              index_metadata=index_metadata
            ).upsert(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings
            )
    
       
    def query_index(self, 
                    name: str,
                    query: str | list[str] | Embedding | list[Embedding] | Embeddings,
                    n_results: int = 10,
                    where: Optional[dict] = None,
                    where_document: Optional[dict] = None,
                    include: list[Literal["metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
                    vector_db_provider: Optional[VectorDBProvider] = None,                            
                    embedding_provider: Optional[LLMProvider] = None,
                    embedding_model: Optional[str] = None,
                    dimensions: Optional[int] = None,
                    distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                    index_metadata: Optional[dict] = None                    
              ) -> VectorDBQueryResult|list[VectorDBQueryResult]:                     
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        index = self.get_or_create_index(
            name=name,
            vector_db_provider=vector_db_provider,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            index_metadata=index_metadata
        )        

                          
        if (is_str_query := isinstance(query, str)) or (isinstance(query, list) and isinstance(query[0], float)):
            if is_str_query:
                query_text = query # Single string
                query_embedding = None
            else:
                query_text = None
                query_embedding = query # Single Embedding (list of floats)
            return index.query(
                query_text=query_text,
                query_embedding=query_embedding,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )

        query_texts = None
        query_embeddings = None
        if isinstance(query, Embeddings):
            query_embeddings = query.list() # List of Embeddings (RootModel) (list of lists of floats)
        elif isinstance(query, list):
            if isinstance((item_0 := query[0]), list) and isinstance(item_0[0], float):
                query_embeddings = query # List of Embeddings (list of lists of floats)            
            elif isinstance(item_0, str):
                query_texts = query # List of strings
        else:
            raise ValueError(f"Invalid query type: {type(query)}. Must be a str, list of str, Embedding (list[float]), list of Embedding list[list[float]] or Embeddings object returned by embed()")

                    
        return index.query_many(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include
                )
    
    
    def delete_index(self,
                     name: str,
                     vector_db_provider: Optional[VectorDBProvider] = None,
                     ):
        return self.get_vector_db(vector_db_provider).delete_index(name)
    
    
    def delete_indexes(self,
                       names: list[str],
                       vector_db_provider: Optional[VectorDBProvider] = None,
                       ):
          return self.get_vector_db(vector_db_provider).delete_indexes(names)
    

    def get_tool_caller(
            self,
            tool_caller_class_or_instance: Type[ToolCaller]|ToolCaller = ToolCaller,
            tools: Optional[Sequence[ToolInput]] = None,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_caller_kwargs: Optional[dict[str, Any]] = None

    ) -> ToolCaller:
        tool_callables = {**self.tool_callables}
        if tools:
            for tool in tools:
                if isinstance(tool, str):
                    tool = self.tools.get(tool)

                if isinstance(tool, Tool) and tool.callable:
                    tool_callables[tool.name] = tool.callable

        if tool_callables:
            tool_callables.update(tool_callables)
        
        if isinstance(tool_caller_class_or_instance, ToolCaller):
            tool_caller_class_or_instance.set_tool_callables(tool_callables)
            return tool_caller_class_or_instance
        
        return tool_caller_class_or_instance(tool_callables=tool_callables, **(tool_caller_kwargs or {}))


    def get_rag_engine(
            self,
            spec_or_name: RAGSpec | str,
            **kwargs
    ) -> RAGEngine:
        if isinstance(spec_or_name, str):
            if (rag_spec := self.rag_specs.get(spec_or_name)) is None:
                raise ValueError(f"RAG spec '{spec_or_name}' not found in self.rag_specs")
        elif isinstance(spec_or_name, RAGSpec):
            rag_spec = spec_or_name
        elif spec_or_name is None:
            rag_spec = RAGSpec(**kwargs)
        else:
            raise ValueError(
                f"Invalid rag_spec: {spec_or_name}. Must be a RAGSpec object or a string (name of a RAGSpec in self.rag_specs)")
        
        if document_db := rag_spec.document_db_class_or_instance:
            if isinstance(document_db, type):
                document_db = document_db(**rag_spec.document_db_kwargs)

        index = self.get_or_create_index(
            name=rag_spec.index_name,
            vector_db_provider=rag_spec.vector_db_provider,
            embedding_provider=rag_spec.embedding_provider,
            embedding_model=rag_spec.embedding_model,
            dimensions=rag_spec.embedding_dimensions,
            distance_metric=rag_spec.embedding_distance_metric,
            document_db=document_db
        )

        if rag_spec.rerank_provider:
            reranker = self.get_reranker(rag_spec.rerank_provider)
        else:
            reranker = None
        
        return RAGEngine(
            spec=rag_spec,
            retreiver=index,
            reranker=reranker
        )  
    

    def get_function(
            self, 
            spec_or_name: Optional[FuncSpec|str] = None,
            **kwargs
            ) -> AIFunction:
        
        
        if isinstance(spec_or_name, str):
            if (spec := self.func_specs.get(spec_or_name)) is None:
                raise ValueError(f"Function spec '{spec_or_name}' not found in self.func_specs")
        elif isinstance(spec_or_name, FuncSpec):
            spec = spec_or_name.model_copy(update=kwargs)
        elif spec_or_name is None:
            spec = FuncSpec(**kwargs)
        else:
            raise ValueError(
                f"Invalid spec: {spec_or_name}. Must be a EvalSpec object or a string (name of a EvalSpec in self.FUNC_SPECS)")
        
        if not spec.provider:
            spec.provider = self.default_llm_provider

        if spec.tool_caller_class_or_instance:
            tool_caller = self.get_tool_caller(
                spec.tool_caller_class_or_instance, 
                spec.tools, 
                spec.tool_callables, 
                spec.tool_caller_kwargs
            )
        else:
            tool_caller = None

        rag_engine = self.get_rag_engine(spec.rag_spec) if spec.rag_spec else None
        return AIFunction(
            spec=spec, 
            rag_engine=rag_engine,
            get_client=self.get_llm_client,
            parent_tools=self.tools,
            tool_caller=tool_caller,
            provider=spec.provider,
        ) 
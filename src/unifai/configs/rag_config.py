from typing import Any, Callable, Collection, Literal, Optional, ParamSpec, Concatenate, Generic, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, InputP
from ..types.db_results import QueryResult
from ..components.prompt_templates import PromptTemplate, PromptModel

from ._base_configs import ComponentConfig
from .document_loader_config import DocumentLoaderConfig
from .document_chunker_config import DocumentChunkerConfig
from .reranker_config import RerankerConfig
from .tokenizer_config import TokenizerConfig
from .vector_db_config import VectorDBConfig, VectorDBCollectionConfig

from pydantic import Field

def leave_query_as_is(*, query: str) -> str:
    return query

class RAGPromptModel(PromptModel):
    result: QueryResult

    def __init__(self, result: QueryResult, *args, **kwargs):
        kwargs["result"] = result
        super().__init__(*args, **kwargs)

    def __call__(
        self, 
        result: QueryResult, 
        *args,
        **kwargs
        ) -> str:
        return super().__call__(*args, **kwargs, result=result)

    
class DefaultRAGPrompt(RAGPromptModel):
    "{query}\n\nCONTEXT:\n{result}"
    query: str

    value_formatters={ 
        QueryResult: lambda result: "\n".join(f"DOCUMENT: {doc.id}\n{doc.text}\n" for doc in result)
    }    
    
    def __init__(self, result: QueryResult, *args, **kwargs):
        super().__init__(*args, **kwargs, result=result)

    def __call__(
        self, 
        result: QueryResult, 
        *args,
        **kwargs
        ) -> str:
        return super().__call__(*args, **kwargs, result=result)   




class RAGPromptTemplate(PromptTemplate):
    def __call__(
        self, 
        result: QueryResult, 
        *args,
        **kwargs
        ) -> str:
        return super().__call__(*args, **kwargs, result=result)
    

default_rag_prompt_template = RAGPromptTemplate(
        template="{query}\n\nCONTEXT:\n{result}",
        value_formatters={ 
            QueryResult: lambda result: "\n".join(f"DOCUMENT: {doc.id}\n{doc.text}\n" for doc in result)
        },
)

class RAGConfig(ComponentConfig, Generic[InputP]):
    component_type: ClassVar = "ragpipe"
    provider: ClassVar[str] = "default"

    document_loader: Optional[DocumentLoaderConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    document_chunker: Optional[DocumentChunkerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    # document_db: Optional[DocumentDBConfig | ProviderName | tuple[ProviderName, ConfigName]] = None
    vector_db: Optional[VectorDBCollectionConfig | VectorDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    reranker: Optional[RerankerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    tokenizer: Optional[TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    
    top_k: int = 20
    top_n: Optional[int] = 10
    where: Optional[dict] = None
    where_document: Optional[dict] = None   
    
    query_modifier: Callable[InputP, str|Callable[..., str]] = Field(default=leave_query_as_is)
    prompt_template: Callable[Concatenate[QueryResult, InputP], str|Callable[..., str]] | Callable[..., str|Callable[..., str]]= Field(default=DefaultRAGPrompt)    


    reranker_model: Optional[ModelName] = None    
    tokenizer_model: Optional[ModelName] = None 

    max_distance: Optional[float] = None
    min_similarity: Optional[float] = None
    max_result_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    use_remaining_documents_to_fill: bool = True

    extra_kwargs: Optional[dict[Literal["load", "chunk", "embed", "upsert", "query", "rerank", "count_tokens"], dict[str, Any]]] = None


class RAGIngestorConfig(ComponentConfig, Generic[InputP]):
    component_type: ClassVar = "rag_ingestor"
    provider: ClassVar[str] = "default"

    document_loader: Optional[DocumentLoaderConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    document_chunker: Optional[DocumentChunkerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    # document_db: Optional[DocumentDBConfig | ProviderName | tuple[ProviderName, ConfigName]] = None
    vector_db: Optional[VectorDBCollectionConfig | VectorDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    tokenizer: Optional[TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
       
    tokenizer_model: Optional[ModelName] = None
    extra_kwargs: Optional[dict[Literal["load", "chunk", "embed", "upsert", "count_tokens"], dict[str, Any]]] = None




class RAGPrompterConfig(ComponentConfig, Generic[InputP]):
    component_type: ClassVar = "rag_prompter"
    provider: ClassVar[str] = "default"

    # document_db: Optional[DocumentDBConfig | ProviderName | tuple[ProviderName, ConfigName]] = None
    vector_db: Optional[VectorDBCollectionConfig | VectorDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    reranker: Optional[RerankerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    tokenizer: Optional[TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    
    reranker_model: Optional[ModelName] = None    
    tokenizer_model: Optional[ModelName] = None 

    top_k: int = 20
    top_n: Optional[int] = 10
    where: Optional[dict] = None
    where_document: Optional[dict] = None   

    # query_modifier: Callable[InputP, str|Callable[InputP, str]] = Field(default=leave_query_as_is)
    # prompt_template: Callable[Concatenate[QueryResult, InputP], str|Callable[Concatenate[QueryResult, InputP], str]] = Field(default=DefaultRAGPrompt)
    query_modifier: Callable[InputP, str|Callable[..., str]] = Field(default=leave_query_as_is)
    prompt_template: Callable[Concatenate[QueryResult, InputP], str|Callable[..., str]] | Callable[..., str|Callable[..., str]]= Field(default=DefaultRAGPrompt)    

    max_distance: Optional[float] = None
    min_similarity: Optional[float] = None
    max_result_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    use_remaining_documents_to_fill: bool = True

    extra_kwargs: Optional[dict[Literal["load", "chunk", "embed", "upsert", "query", "rerank", "count_tokens"], dict[str, Any]]] = None

RAGConfig(prompt_template=PromptTemplate("{query}\n\nCONTEXT:\n{result}"))

custom_rag_prompt_template = RAGPromptTemplate(template="{query}\n\nCONTEXT:\n{result} {extra}")

def add_to_query(*, query: str, extra: str) -> str:
    return f"{query} {extra}"

prompt_config = RAGPrompterConfig(
    vector_db="vector_db",
    reranker="reranker",
    tokenizer="tokenizer",
    # query_modifier=add_to_query,
    prompt_template=custom_rag_prompt_template,
    # prompt_template=DefaultRAGPrompt,
    # query_modifier=leave_query_as_is,
    # prompt_template=PromptTemplate("{query}\n\nCONTEXT:\n{result}"),
)
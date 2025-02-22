from typing import Any, Callable, Collection, Literal, Optional, ParamSpec, Concatenate, Generic, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, InputP as QueryInputP
from ..types.db_results import QueryResult
from ..types.documents import Document, Documents
from ..components.prompt_templates.rag_prompt_model import RAGPromptModel


from ._base_configs import ComponentConfig
from .document_loader_config import DocumentLoaderConfig, LoaderInputP
from .document_chunker_config import DocumentChunkerConfig
from .reranker_config import RerankerConfig
from .tokenizer_config import TokenizerConfig
from .vector_db_config import VectorDBConfig, VectorDBCollectionConfig

from pydantic import Field

def leave_query_as_is(query: str) -> str:
    return query

def load_documents_as_is(documents: Iterable[Document]) -> Iterable[Document]:
    return documents

class RAGConfig(ComponentConfig, Generic[LoaderInputP, QueryInputP]):
    component_type: ClassVar = "ragpipe"
    provider: ClassVar[str] = "default"

    # document_loader: Optional[DocumentLoaderConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    document_loader: Callable[LoaderInputP, Iterable[Document]] | DocumentLoaderConfig[LoaderInputP] | ProviderName | tuple[ProviderName, ComponentName] = Field(default=load_documents_as_is)

    document_chunker: Optional[DocumentChunkerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    # document_db: Optional[DocumentDBConfig | ProviderName | tuple[ProviderName, ConfigName]] = None
    vector_db: Optional[VectorDBCollectionConfig | VectorDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    reranker: Optional[RerankerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    tokenizer: Optional[TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"

    query_modifier: Callable[QueryInputP, str|Callable[..., str]] = Field(default=leave_query_as_is)
    prompt_template: Callable[Concatenate[QueryResult, QueryInputP], str|Callable[..., str]] | Callable[..., str|Callable[..., str]]= Field(default=RAGPromptModel)   

    top_k: int = 20
    top_n: Optional[int] = 10
    where: Optional[dict] = None
    where_document: Optional[dict] = None   
    
    reranker_model: Optional[ModelName] = None
    tokenizer_model: Optional[ModelName] = None

    max_distance: Optional[float] = None
    min_similarity: Optional[float] = None
    max_result_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    use_remaining_documents_to_fill: bool = True

    extra_kwargs: Optional[dict[Literal["load", "chunk", "embed", "upsert", "query", "rerank", "count_tokens"], dict[str, Any]]] = None
from typing import Any, Callable, Collection, Literal, Optional, ParamSpec, Concatenate, Generic, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, InputP as QueryInputP
from ..types.db_results import QueryResult
from ..types.documents import Document, Documents
from ..components.prompt_templates.rag_prompt_model import RAGPromptModel
from ..components._base_components._base_document_loader import DocumentLoader
from ..components._base_components._base_document_db import DocumentDB
from ..components._base_components._base_document_chunker import DocumentChunker
from ..components._base_components._base_vector_db import VectorDB, VectorDBCollection
from ..components._base_components._base_reranker import Reranker
from ..components._base_components._base_tokenizer import Tokenizer

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

    document_loader: Callable[LoaderInputP, Iterable[Document]] | DocumentLoaderConfig[LoaderInputP] | ProviderName | tuple[ProviderName, ComponentName] = Field(default=load_documents_as_is)
    document_chunker: Optional[DocumentChunker | DocumentChunkerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"

    vector_db: Optional[VectorDB | VectorDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    vector_db_collection: Optional[VectorDBCollection | VectorDBCollectionConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"

    reranker: Optional[Reranker | RerankerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    reranker_model: Optional[ModelName] = None

    tokenizer: Optional[Tokenizer | TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"
    tokenizer_model: Optional[ModelName] = None

    query_modifier: Callable[QueryInputP, str|Callable[..., str]] = Field(default=leave_query_as_is)
    prompt_template: Callable[Concatenate[QueryResult, QueryInputP], str|Callable[..., str]] | Callable[..., str|Callable[..., str]]= Field(default=RAGPromptModel)   

    top_k: int = 20
    top_n: Optional[int] = 10
    where: Optional[dict] = None
    where_document: Optional[dict] = None   
    
    max_distance: Optional[float] = None
    min_similarity: Optional[float] = None
    max_result_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    use_remaining_documents_to_fill: bool = True

    extra_kwargs: Optional[dict[Literal["load", "chunk", "embed", "upsert", "query", "rerank", "count_tokens"], dict[str, Any]]] = None

RAGConfig(
    
)
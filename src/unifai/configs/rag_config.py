from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName
from ..types.db_results import QueryResult
from ..components.prompt_template import PromptTemplate

from ._base_configs import ComponentConfig
from .document_loader_config import DocumentLoaderConfig
from .document_chunker_config import DocumentChunkerConfig
from .reranker_config import RerankerConfig
from .tokenizer_config import TokenizerConfig
from .vector_db_config import VectorDBConfig, VectorDBCollectionConfig

class RAGConfig(ComponentConfig):
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
    query_modifier: Optional[Callable[[str], str]] = None


    reranker_model: Optional[ModelName] = None    
    tokenizer_model: Optional[ModelName] = None 

    prompt_template: PromptTemplate = PromptTemplate(
        "{query}\n\nCONTEXT:\n{result}",
        value_formatters={ 
            QueryResult: lambda result: "\n".join(f"DOCUMENT: {doc.id}\n{doc.text}\n" for doc in result)
        },
    )
    prompt_template_kwargs: Optional[dict[str, Any]] = None

    max_distance: Optional[float] = None
    min_similarity: Optional[float] = None
    max_result_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    use_remaining_documents_to_fill: bool = True

    extra_kwargs: Optional[dict[Literal["load", "chunk", "embed", "upsert", "query", "rerank", "count_tokens"], dict[str, Any]]] = None

RAGConfig(prompt_template=PromptTemplate("{query}\n\nCONTEXT:\n{result}"))
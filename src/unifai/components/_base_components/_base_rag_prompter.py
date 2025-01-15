from typing import TYPE_CHECKING, Any, Callable, ParamSpec, Concatenate, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator, NoReturn, Generic, TypeVar

if TYPE_CHECKING:
    from ...types.annotations import ComponentName, ModelName, ProviderName

from .._base_components._base_document_chunker import DocumentChunker
from .._base_components._base_document_db import DocumentDB, DocumentDBCollection
from .._base_components._base_document_loader import DocumentLoader
from .._base_components._base_embedder import Embedder
from .._base_components._base_vector_db import VectorDB, VectorDBCollection
from .._base_components._base_reranker import Reranker
from .._base_components._base_tokenizer import Tokenizer
from ._base_prompt_template import PromptTemplate

from ...configs.rag_config import RAGPrompterConfig, InputP
from ...configs.document_chunker_config import DocumentChunkerConfig
from ...configs.document_db_config import DocumentDBConfig, DocumentDBCollectionConfig
from ...configs.document_loader_config import DocumentLoaderConfig
from ...configs.embedder_config import EmbedderConfig
from ...configs.reranker_config import RerankerConfig
from ...configs.tokenizer_config import TokenizerConfig
from ...configs.vector_db_config import VectorDBConfig, VectorDBCollectionConfig

from ...types import Document, Documents, ResponseInfo, QueryResult

from ...utils import chunk_iterable, combine_dicts

from .__base_component import UnifAIComponent

RAGPrompterConfigT = TypeVar('RAGPrompterConfigT', bound=RAGPrompterConfig)

class BaseRAGPrompter(UnifAIComponent[RAGPrompterConfigT], Generic[RAGPrompterConfigT, InputP]):
    component_type = "rag_prompter"
    provider = "base"

    can_get_components = True

    def _setup(self) -> None:
        super()._setup()
        self._vector_db = self.init_kwargs.get("vector_db")
        self._vector_db_collection = self.init_kwargs.get("vector_db_collection")
        self._reranker = self.init_kwargs.get("reranker")
        self._tokenizer = self.init_kwargs.get("tokenizer")
        self._extra_kwargs = self.config.extra_kwargs or {}
        self.response_infos: list[ResponseInfo] = []
    
    @property
    def vector_db(self) -> VectorDB:
        if self._vector_db is not None:
            return self._vector_db        
        if not self.config.vector_db:
            raise ValueError("vector_db is required. Set vector_db in the config or provide an instance at runtime.")
        if isinstance(self.config.vector_db, VectorDBCollectionConfig):
            self._vector_db = self._get_component("vector_db", self.config.vector_db.provider)
        else:       
            self._vector_db = self._get_component("vector_db", self.config.vector_db)
        return self._vector_db

    @property
    def vector_db_collection(self) -> VectorDBCollection:
        if self._vector_db_collection is not None:
            return self._vector_db_collection        
        if not self.config.vector_db:
            raise ValueError("vector_db is required. Set vector_db in the config or provide an instance at runtime.")
        if isinstance(self.config.vector_db, VectorDBCollectionConfig):
            self._vector_db_collection = self.vector_db.get_or_create_collection_from_config(self.config.vector_db)
        else:
            self._vector_db_collection = self.vector_db.get_or_create_collection()
        return self._vector_db_collection

    @vector_db_collection.setter
    def vector_db_collection(self, value: "Optional[VectorDBCollection | VectorDBCollectionConfig | ProviderName | tuple[ProviderName, ComponentName]]") -> None:
        if value is None or isinstance(value, VectorDBCollection):
            self._vector_db_collection = value
        else:
            self._vector_db_collection = self._get_component("vector_db", value)
    
    @property
    def reranker(self) -> Optional[Reranker]:
        if self._reranker is not None:
            return self._reranker
        if self.config.reranker:
            self._reranker = self._get_component("reranker", self.config.reranker)
        return self._reranker
    
    @reranker.setter
    def reranker(self, value: "Optional[Reranker | RerankerConfig | ProviderName | tuple[ProviderName, ComponentName]]") -> None:
        if value is None or isinstance(value, Reranker):
            self._reranker = value
        else:
            self._reranker = self._get_component("reranker", value)    
    
    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is not None:
            return self._tokenizer        
        if not self.config.tokenizer:
            raise ValueError("tokenizer is required. Set tokenizer in the config or provide an instance at runtime.")
        return self._get_component("tokenizer", self.config.tokenizer)
    
    @tokenizer.setter
    def tokenizer(self, value: "Optional[Tokenizer | TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]]") -> None:
        if value is None or isinstance(value, Tokenizer):
            self._tokenizer = value
        else:
            self._tokenizer = self._get_component("tokenizer", value)
        
        
    def query(
            self, 
            top_k: Optional[int] = None,
            top_n: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None, 
            max_distance: Optional[float] = None,
            min_similarity: Optional[float] = None,                    
            query_modifier: Optional[Callable[InputP, str]] = None,
            query_kwargs: Optional[dict] = None,
            reranker_model: Optional["ModelName"] = None,          
            rerank_kwargs: Optional[dict] = None,
            *args: InputP.args,
            **kwargs: InputP.kwargs
            ) -> QueryResult:
        
        top_k = top_k or self.config.top_k
        top_n = top_n or self.config.top_n
        where = where or self.config.where
        where_document = where_document or self.config.where_document
        max_distance = max_distance or self.config.max_distance
        min_similarity = min_similarity or self.config.min_similarity

        _query_modifier = query_modifier or self.config.query_modifier
        query_input = _query_modifier(*args, **kwargs)
        if callable(query_input):
            query_input = query_input(*args, **kwargs)

        query_kwargs = combine_dicts(self._extra_kwargs.get("query"), query_kwargs)
        query_result = self.vector_db_collection.query(query_input, top_k, where, where_document, **query_kwargs)
        
        if max_distance is not None:
            # TODO: maybe move to VectorDBCollection
            query_result.trim_by_distance(max_distance)

        # If a reranker is configured and there are results rerank them
        if self.reranker and query_result.ids:
            reranker_model = reranker_model or self.config.reranker_model
            rerank_kwargs = combine_dicts(self._extra_kwargs.get("rerank"), rerank_kwargs)
            query_result = self.reranker.rerank(
                query=query_input,
                query_result=query_result,
                top_n=top_n,
                model=reranker_model,
                **rerank_kwargs
            )
            if min_similarity is not None:
                query_result.trim_by_similarity_score(min_similarity)

        return query_result

    def construct_rag_prompt(
            self,
            query_result: QueryResult,
            prompt_template: Optional[Callable[Concatenate[QueryResult, InputP], str]] = None,
            *args: InputP.args,
            **kwargs: InputP.kwargs
            ) -> str:
        
        _prompt_template = prompt_template or self.config.prompt_template
        # kwargs["query"] = query_result.query
        _kwargs = combine_dicts({"query": query_result.query}, kwargs)
        prompt = _prompt_template(query_result, *args, **_kwargs)
        if callable(prompt):
            prompt = prompt(query_result, *args, **kwargs)
        return prompt

    def prompt(
            self, 
            top_k: Optional[int] = None,
            top_n: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            max_distance: Optional[float] = None,
            min_similarity: Optional[float] = None,            
            query_modifier: Optional[Callable[InputP, str]] = None,
            query_kwargs: Optional[dict] = None,
            reranker_model: Optional[str] = None,
            rerank_kwargs: Optional[dict] = None,
            prompt_template: Optional[Callable[Concatenate[QueryResult, InputP], str]] = None,
            *args: InputP.args,
            **kwargs: InputP.kwargs
            ) -> str:

        query_result = self.query(top_k, top_n, where, where_document, max_distance, min_similarity, query_modifier, query_kwargs, reranker_model, rerank_kwargs, *args, **kwargs)
        return self.construct_rag_prompt(query_result, prompt_template, *args, **kwargs)
    
    __call__ = prompt
from typing import Type, TypeVar, Generic, Optional, Any, Literal
from abc import abstractmethod

from ...types import Embedding, Embeddings, QueryResult, RankedDocument
from ...configs._base_configs import ComponentConfig
from .__base_component import UnifAIComponent

RetreiverConfig = TypeVar("RetreiverConfig", bound=ComponentConfig)
QueryInputT = TypeVar("QueryInputT")
QueryManyInputT = TypeVar("QueryManyInputT")

class Retriever(UnifAIComponent[RetreiverConfig], Generic[RetreiverConfig, QueryInputT, QueryManyInputT]):
    component_type = "retreiver"
    provider = "base"
    config_class: Type[RetreiverConfig]
    can_get_components = False

    # Abstract Methods
    @abstractmethod
    def _query(
            self,              
            query_input: QueryInputT,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
            ) -> QueryResult:        
        ...

    @abstractmethod
    def _query_documents(
            self,              
            query_input: QueryInputT,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
            ) -> list[RankedDocument]:        
        ...

    @abstractmethod
    def _query_many(
            self,              
            query_inputs: QueryManyInputT,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
            ) -> list[QueryResult]:        
        ...


    # Concrete Methods
    def query(
            self,              
            query_input: QueryInputT,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
            ) -> QueryResult:        
        return self._run_func(self._query, query_input, top_k, where, where_document, include, **kwargs)

    def query_documents(
            self,              
            query_input: QueryInputT,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
            ) -> list[RankedDocument]:        
        return self._run_func(self._query_documents, query_input, top_k, where, where_document, include, **kwargs)

    def query_many(
            self,              
            query_inputs: QueryManyInputT,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
            ) -> list[QueryResult]:        
        return self._run_func(self._query_many, query_inputs, top_k, where, where_document, include, **kwargs)

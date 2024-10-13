from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from ..adapters._base_vector_db_index import VectorDBIndex
from ..adapters._base_reranker import Reranker
from ..types.rag_spec import RAGSpec, VectorDBQueryResult
from .prompt_template import PromptTemplate


class Retriever:
    def query(self, query_text: str, n_results: int, **kwargs) -> VectorDBQueryResult:
        raise NotImplementedError


class RAGEngine:

    def __init__(
            self, 
            rag_spec: RAGSpec,
            retreiver: VectorDBIndex|Retriever,
            reranker: Optional[Reranker] = None,
        ):
        self.retreiver = retreiver
        self.reranker = reranker
        self.rag_spec = rag_spec


    def retrieve(
            self, 
            query: str, 
            top_k: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **retreiver_kwargs
        ) -> VectorDBQueryResult:

        n_results = top_k or self.rag_spec.top_k or self.rag_spec.top_n
        where = where or self.rag_spec.where
        where_document = where_document or self.rag_spec.where_document
        retreiver_kwargs = {**self.rag_spec.retreiver_kwargs, **retreiver_kwargs}
        return self.retreiver.query(
            query_text=query,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **retreiver_kwargs
        )


    def rerank(
            self, 
            query: str, 
            query_result: VectorDBQueryResult,
            model: Optional[str] = None,
            top_n: Optional[int] = None,
            **reranker_kwargs
            ) -> VectorDBQueryResult:
        if self.reranker is None:
            # No reranker just return query_result as is
            return query_result
        
        model = model or self.rag_spec.rerank_model
        top_n = top_n or self.rag_spec.top_n
        reranker_kwargs = {**self.rag_spec.reranker_kwargs, **reranker_kwargs}
        return self.reranker.rerank(
            query=query,
            query_result=query_result,
            model=model,
            top_n=top_n,
            **reranker_kwargs
        )
    
    
    def query(
            self, 
            query: str,
            retreiver_kwargs: Optional[dict] = None,
            reranker_kwargs: Optional[dict] = None,
            ) -> VectorDBQueryResult:
        query_result = self.retrieve(query, **(retreiver_kwargs or {}))
        if self.reranker:
            query_result = self.rerank(query, query_result, **(reranker_kwargs or {}))        
        return query_result


    def augment(
            self,
            query: str,
            query_result: VectorDBQueryResult,
            prompt_template: Optional[PromptTemplate] = None,
            **prompt_template_kwargs
            ) -> str:
        
        prompt_template = prompt_template or self.rag_spec.prompt_template
        prompt_template_kwargs = {**self.rag_spec.prompt_template_kwargs, **prompt_template_kwargs}
        return prompt_template.format(query=query, query_result=query_result, **prompt_template_kwargs)


    def ragify(
            self, 
            query: str,
            prompt_template: Optional[PromptTemplate] = None,
            retreiver_kwargs: Optional[dict] = None,
            reranker_kwargs: Optional[dict] = None,
            prompt_template_kwargs: Optional[dict] = None,            
            ) -> str:

        query_result = self.query(query, retreiver_kwargs, reranker_kwargs)
        return self.augment(query, query_result, prompt_template, **(prompt_template_kwargs or {}))
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from ..components.retrievers._base_retriever import Retriever
from ..components.rerankers._base_reranker import Reranker
from ..types.vector_db import VectorDBQueryResult
from ..components.prompt_template import PromptTemplate
from .specs import RAGSpec


class RAGEngine:

    def __init__(
            self, 
            spec: RAGSpec,
            retriever: Retriever,
            reranker: Optional[Reranker] = None,
        ):
        self.retriever = retriever
        self.reranker = reranker
        self.spec = spec


    def retrieve(
            self, 
            query: str, 
            top_k: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **retriever_kwargs
        ) -> VectorDBQueryResult:

        n_results = top_k or self.spec.top_k or self.spec.top_n
        where = where or self.spec.where
        where_document = where_document or self.spec.where_document
        retriever_kwargs = {**self.spec.retriever_kwargs, **retriever_kwargs}
        return self.retriever.query(
            query_text=query,
            n_results=n_results,
            where=where,
            where_document=where_document,
            **retriever_kwargs
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
        
        model = model or self.spec.rerank_model
        top_n = top_n or self.spec.top_n
        reranker_kwargs = {**self.spec.reranker_kwargs, **reranker_kwargs}
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
        
        prompt_template = prompt_template or self.spec.prompt_template
        prompt_template_kwargs = {**self.spec.prompt_template_kwargs, **prompt_template_kwargs}
        return prompt_template.format(query=query, query_result=query_result, **prompt_template_kwargs)


    def ragify(
            self, 
            query: str,
            prompt_template: Optional[PromptTemplate] = None,
            retriever_kwargs: Optional[dict] = None,
            reranker_kwargs: Optional[dict] = None,
            **prompt_template_kwargs
            ) -> str:

        query_result = self.query(query, retriever_kwargs, reranker_kwargs)
        return self.augment(query, query_result, prompt_template, **prompt_template_kwargs)
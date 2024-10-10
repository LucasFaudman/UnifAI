from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai.types import (
    LLMProvider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    ToolCall,
    Usage,
    VectorDBQueryResult,
)
from unifai.type_conversions import standardize_tools, standardize_messages, standardize_tool_choice, standardize_response_format
from .prompt_template import PromptTemplate

from unifai.wrappers._base_vector_db_index import VectorDBIndex
from unifai.wrappers._base_reranker_client import RerankerClient
from unifai.types.rag_spec import RAGSpec

from pydantic import BaseModel

class Retriever:
    def query(self, query_text: str, n_results: int, **kwargs) -> VectorDBQueryResult:
        raise NotImplementedError


# class IndexOnlyRetriever(Retriever):
#     def __init__(self, index: VectorDBIndex):
#         self.index = index

#     def query(self, query_text: str, n_results: int, **kwargs) -> VectorDBQueryResult:
#         return self.index.query(query_text=query_text, n_results=n_results, **kwargs)


class Docloader:
    def __call__(self, ids: Collection[str]) -> list[str]:
        raise NotImplementedError   


class IndexDocloaderRetriever(Retriever):
    def __init__(
            self,
            index: VectorDBIndex, 
            docloader: Callable[[Collection[str]], list[str]]
        ):
        self.index = index
        self.docloader = docloader

    def query(self, query_text: str, n_results: int, **kwargs) -> VectorDBQueryResult:
        query_result = self.index.query(query_text=query_text, n_results=n_results, **kwargs)
        query_result.documents = self.docloader(query_result.ids)
        return query_result


  



class RAGEngine:

    def __init__(
            self, 
            rag_spec: RAGSpec,
            retreiver: VectorDBIndex|Retriever,
            reranker: Optional[RerankerClient] = None,
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


    def ragify(self, 
            query: str,
            prompt_template: Optional[PromptTemplate] = None,
            retreiver_kwargs: Optional[dict] = None,
            reranker_kwargs: Optional[dict] = None,
            prompt_template_kwargs: Optional[dict] = None,            
            ) -> str:

        query_result = self.retrieve(query, **(retreiver_kwargs or {}))
        if self.reranker:
            query_result = self.rerank(query, query_result, **(reranker_kwargs or {}))
        return self.augment(query, query_result, prompt_template, **(prompt_template_kwargs or {}))
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai.types import (
    AIProvider, 
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



class RAGEngine:

    @staticmethod
    def default_query_result_formatter(result: VectorDBQueryResult) -> str:
        if result.documents is None:
            raise ValueError("Result must have documents to format")
        return "\n".join(f"DOCUMENT: {id}\n{document}" for id, document in zip(result.ids, result.documents))

    default_prompt_template = PromptTemplate(
        "{prompt_header}{query}{sep}{result_header}{query_result}{response_start}",
        prompt_header="PROMPT: ",
        result_header="CONTEXT: ",
        sep="\n\n",
        response_start="\n\nRESPONSE: ",
        value_formatters={VectorDBQueryResult: default_query_result_formatter},
    )          

    def __init__(self, 
                 index: VectorDBIndex,
                 prompt_template: PromptTemplate = default_prompt_template,
                 retreiver_kwargs: Optional[dict] = None,
                 reranker_kwargs: Optional[dict] = None,
                 prompt_template_kwargs: Optional[dict] = None,
                 ):
        self.index = index
        self.prompt_template = prompt_template
        self.retreiver_kwargs = retreiver_kwargs or {}
        self.reranker_kwargs = reranker_kwargs or {}
        self.prompt_template_kwargs = prompt_template_kwargs or {}


    def retrieve(
            self, 
            query: str, 
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **retreiver_kwargs) -> VectorDBQueryResult:
        return self.index.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            where_document=where_document,
            **retreiver_kwargs
        )[0]


    def rerank(
            self, 
            query: str, 
            query_result: VectorDBQueryResult,
            top_n: Optional[int] = None,
            **reranker_kwargs) -> VectorDBQueryResult:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    

    def augment(
            self,
            query: str,
            query_result: VectorDBQueryResult,
            prompt_template: PromptTemplate,
            **prompt_template_kwargs) -> str:
        return prompt_template.format(query=query, query_result=query_result, **prompt_template_kwargs)
    
  
    def retrieve_and_augment(
            self,
            query: str,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            prompt_template: Optional[PromptTemplate] = None,
            **prompt_template_kwargs) -> str:
        query_result = self.retrieve(query, top_k, where, where_document, **prompt_template_kwargs)
        return self.augment(query, query_result, prompt_template or self.default_prompt_template, **prompt_template_kwargs)
            

    def retreive_rerank_and_augment(
            self,
            query: str,
            top_k: int = 10,
            top_n: int = 5,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            retreiver_kwargs: Optional[dict] = None,
            reranker_kwargs: Optional[dict] = None,
            prompt_template: Optional[PromptTemplate] = None,
            **prompt_template_kwargs) -> str:
        query_result = self.retrieve(query, top_k, where, where_document, **(retreiver_kwargs or {}))
        reranked_query_result = self.rerank(query, query_result, top_n, **(reranker_kwargs or {}))
        return self.augment(query, reranked_query_result, prompt_template or self.default_prompt_template, **prompt_template_kwargs)
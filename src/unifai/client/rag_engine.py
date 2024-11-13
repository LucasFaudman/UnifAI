from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator
from pydantic import BaseModel, Field, ConfigDict


from ..components.retrievers._base_retriever import Retriever
from ..components.rerankers._base_reranker import Reranker
from ..types.vector_db import VectorDBQueryResult
from ..components.prompt_template import PromptTemplate

class RAGConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_db_provider: Optional[str] = None
    index_name: str = "default_index"
    query_kwargs: dict[str, Any] = Field(default_factory=dict)
    
    document_db_provider: Optional[str] = None
    document_db_get_kwargs: dict[str, Any] = Field(default_factory=dict)

    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedding_distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None
    
    rerank_provider: Optional[str] = None
    rerank_model: Optional[str] = None
    rerank_kwargs: dict[str, Any] = Field(default_factory=dict)

    top_k: Optional[int] = 20
    top_n: Optional[int] = 10
    where: Optional[dict] = None
    where_document: Optional[dict] = None

    prompt_template: PromptTemplate = PromptTemplate(
        "{prompt_header}{query}{sep}{result_header}{query_result}{response_start}",
        value_formatters={
            VectorDBQueryResult: 
            lambda result: "\n".join(f"DOCUMENT: {id}\n{doc}" for id, doc in result.zip("ids", "documents"))
        },
        prompt_header="PROMPT:\n",
        result_header="CONTEXT:\n",        
        response_start="\n\nRESPONSE: ",
        sep="\n\n",        
    )
    prompt_template_kwargs: dict[str, Any] = Field(default_factory=dict)

class RAGEngine:

    def __init__(
            self, 
            config: RAGConfig,
            retriever: Retriever,
            reranker: Optional[Reranker] = None,
        ):
        self.config = config
        self.retriever = retriever
        self.reranker = reranker

    def retrieve(
            self, 
            query: str, 
            top_k: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **query_kwargs
        ) -> VectorDBQueryResult:
        return self.retriever.query(
            query_text=query,
            **{
            **self.config.query_kwargs, 
            **query_kwargs, 
            "top_k": top_k or self.config.top_k, 
            "where": where or self.config.where, 
            "where_document": where_document or self.config.where_document, 
            }
        )

    def rerank(
            self, 
            query: str, 
            query_result: VectorDBQueryResult,
            top_n: Optional[int] = None,
            rerank_model: Optional[str] = None,
            **rerank_kwargs
            ) -> VectorDBQueryResult:
        
        top_n = top_n or self.config.top_n #or self.config.top_k
        if self.reranker is None:
            # No reranker just return query_result as is reduced to top_n if specified
            return query_result.reduce_to_top_n(top_n) if top_n else query_result
        
        return self.reranker.rerank(
            query=query,
            query_result=query_result,
            **{
            **self.config.rerank_kwargs, 
            **rerank_kwargs, 
            "top_n": top_n, 
            "model": rerank_model or self.config.rerank_model
            }
        )
    
    def construct_rag_prompt(
            self,
            query: str,
            query_result: VectorDBQueryResult,
            prompt_template: Optional[PromptTemplate] = None,
            **prompt_template_kwargs
            ) -> str:
        
        prompt_template = prompt_template or self.config.prompt_template
        prompt_template_kwargs = {**self.config.prompt_template_kwargs, **prompt_template_kwargs}
        return prompt_template.format(query=query, query_result=query_result, **prompt_template_kwargs)

    def query(
            self, 
            query: str,
            top_k: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            top_n: Optional[int] = None,
            rerank_model: Optional[str] = None,                        
            query_kwargs: Optional[dict] = None,
            rerank_kwargs: Optional[dict] = None,
            ) -> VectorDBQueryResult:
        query_result = self.retrieve(query, top_k, where, where_document, **{**self.config.query_kwargs, **(query_kwargs or {})})
        return self.rerank(query, query_result, top_n, rerank_model, **{**self.config.rerank_kwargs, **(rerank_kwargs or {})})
    
    def ragify(
            self, 
            query: str,
            top_k: Optional[int] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            top_n: Optional[int] = None,
            rerank_model: Optional[str] = None,                        
            query_kwargs: Optional[dict] = None,
            rerank_kwargs: Optional[dict] = None,
            prompt_template: Optional[PromptTemplate] = None,
            **prompt_template_kwargs
            ) -> str:

        query_result = self.query(query, top_k, where, where_document, top_n, rerank_model, query_kwargs, rerank_kwargs)
        return self.construct_rag_prompt(query, query_result, prompt_template, **prompt_template_kwargs)

    # def query(
    #         self, 
    #         query: str,
    #         query_kwargs: Optional[dict] = None,
    #         rerank_kwargs: Optional[dict] = None,
    #         ) -> VectorDBQueryResult:
    #     query_result = self.retrieve(query, **(query_kwargs or {}))
    #     if self.reranker:
    #         query_result = self.rerank(query, query_result, **(rerank_kwargs or {}))        
    #     return query_result
    

    # def ragify(
    #         self, 
    #         query: str,
    #         query_kwargs: Optional[dict] = None,
    #         rerank_kwargs: Optional[dict] = None,
    #         prompt_template: Optional[PromptTemplate] = None,
    #         **prompt_template_kwargs
    #         ) -> str:

    #     query_result = self.query(query, query_kwargs, rerank_kwargs)
    #     return self.construct_rag_prompt(query, query_result, prompt_template, **prompt_template_kwargs)
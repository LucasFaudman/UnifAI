from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator



from unifai.wrappers._base_vector_db_index import VectorDBIndex
from unifai.wrappers._base_reranker_client import RerankerClient
from .valid_inputs import EmbeddingProvider, VectorDBProvider, RerankProvider
from .vector_db import VectorDBQueryResult

from unifai.unifai_client.prompt_template import PromptTemplate


from pydantic import BaseModel, Field

def default_query_result_formatter(result: VectorDBQueryResult) -> str:
    if result.documents is None:
        raise ValueError("Result must have documents to format")
    return "\n".join(f"DOCUMENT: {id}\n{document}" for id, document in zip(result.ids, result.documents))

DEFAULT_RAG_PROMPT_TEMPLATE = PromptTemplate(
    "{prompt_header}{query}{sep}{result_header}{query_result}{response_start}",
    prompt_header="PROMPT:\n",
    result_header="CONTEXT:\n",
    sep="\n\n",
    response_start="\n\nRESPONSE: ",
    value_formatters={VectorDBQueryResult: default_query_result_formatter},
)       

class RAGSpec(BaseModel):
    name: str = "default"
    index_name: str
    prompt_template: PromptTemplate = DEFAULT_RAG_PROMPT_TEMPLATE
    top_n: int = 5
    top_k: Optional[int] = 10    
    where: Optional[dict] = None
    where_document: Optional[dict] = None


    vector_db_provider: Optional[VectorDBProvider] = None                            
    embedding_provider: Optional[EmbeddingProvider] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedding_distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None
    docloader: Optional[Callable[[Collection[str]],list[str]]] = None

    rerank_provider: Optional[RerankProvider] = None
    rerank_model: Optional[str] = None
        
    retreiver_kwargs: dict[str, Any] = Field(default_factory=dict)
    reranker_kwargs: dict[str, Any] = Field(default_factory=dict)
    prompt_template_kwargs: dict[str, Any] = Field(default_factory=dict)
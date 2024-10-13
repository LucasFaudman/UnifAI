from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_adapter import BaseAdapter, UnifAIExceptionConverter
from ._base_embedder import Embedder
from ._base_llm_client import LLMClient
from ._base_reranker import Reranker

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError

T = TypeVar("T")

class RankBM25Wrapper(Reranker):
    provider = "rank_bm25"
    default_reranking_model = "BM25Okapi"
    

    def import_client(self):
        import rank_bm25
        return rank_bm25
    

    def init_client(self, **client_kwargs):
        self.client_kwargs.update(client_kwargs)        
        # DO NOT set self._client to prevent issues pickling the module
        # return self.import_client()        
        self._client = self.import_client()
        return self._client
        

    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: Exception) -> UnifAIError:
        return UnifAIError(message=str(exception), original_exception=exception)
    

    # List Models
    def list_models(self) -> list[str]:
        return ["BM25", "BM25Okapi", "BM25L", "BM25Plus"]
    

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    
    # Reranking
    def _get_rerank_response(
        self,
        query: str,
        query_result: VectorDBQueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:

        if not query_result.documents:
            raise ValueError("Cannot rerank an empty query result")

        if not (algo_cls := getattr(self.client, model, None)):
            raise ValueError(f"Invalid BM25 model: {model}. Must be one of {self.list_models()}")

        tokenized_documents = [self.tokenize(doc) for doc in query_result.documents]
        bm25 = algo_cls(
            corpus=tokenized_documents,
            # tokenizer=self.tokenize, # Setting tokenizer uses multiprocessing.Pool 
            **kwargs
        )
        return bm25.get_scores(self.tokenize(query))


    def _extract_reranked_order(
        self,
        response: Any,
        top_n: Optional[int] = None,        
        **kwargs
        ) -> list[int]:
        
        # return argsort(response, **kwargs)[::-1][:top_n].tolist()
        return [index for index, score in sorted(enumerate(response), key=lambda x: x[1], reverse=True)[:top_n]]


from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import QueryResult
from .._base_components._base_reranker import Reranker

class RankBM25Reranker(Reranker):
    provider = "rank_bm25"
    can_get_components = True
    default_reranking_model = "BM25Okapi"
    

    def import_client(self):
        import rank_bm25
        return rank_bm25
    

    def init_client(self, **init_kwargs):
        self.init_kwargs.update(init_kwargs)        
        # DO NOT set self._client to prevent issues pickling the module
        # return self.import_client()        
        self._client = self.import_client()
        return self._client
        

    # List Models
    def list_models(self) -> list[str]:
        return ["BM25", "BM25Okapi", "BM25L", "BM25Plus"]
    

    def tokenize(self, text: str) -> list[str]:
        # TODO - Add support for custom tokenization
        return text.split()

    
    # Reranking
    def _get_rerank_response(
        self,
        query: str,
        query_result: QueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:

        if not query_result.texts:
            raise ValueError("Cannot rerank an empty query result")

        if not (algo_cls := getattr(self.client, model, None)):
            raise ValueError(f"Invalid BM25 model: {model}. Must be one of {self.list_models()}")

        tokenized_documents = [self.tokenize(doc) for doc in query_result.texts]
        bm25 = algo_cls(
            corpus=tokenized_documents,
            # tokenizer=self.tokenize, # Setting tokenizer uses multiprocessing.Pool 
            **kwargs
        )
        return bm25.get_scores(self.tokenize(query))
from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self, ClassVar

from .._core._base_adapter import UnifAIAdapter, UnifAIComponent
from .._core._base_embedder import Embedder
from .._core._base_reranker import Reranker

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError

from sentence_transformers import SentenceTransformer, CrossEncoder

T = TypeVar("T")

from importlib import import_module
from itertools import product

class SentenceTransformersAdapter(Embedder, Reranker):
    client: SentenceTransformer

    provider = "sentence_transformers"
    default_embedding_model = "multi-qa-mpnet-base-cos-v1"
    default_reranking_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cache for loaded SentenceTransformer models
    st_model_cache: ClassVar[dict[str, SentenceTransformer]] = {}
    # Cache for loaded CrossEncoder models
    ce_model_cache: ClassVar[dict[str, CrossEncoder]] = {}


    def lazy_import(self, module_name: str) -> Any:
        module_name, *submodules = module_name.split(".")
        if not (module := globals().get(module_name)):
            module = import_module(module_name)
            globals()[module_name] = module
                    
        for submodule in submodules:
            module = getattr(module, submodule)        
        return module


    def import_client(self):
        return self.lazy_import("sentence_transformers")


    def init_client(self, **client_kwargs):
        self.client_kwargs.update(client_kwargs)


    # Convert Exceptions from Client Exception Types to UnifAI Exceptions for easier handling
    def convert_exception(self, exception: Exception) -> UnifAIError:
        return UnifAIError(message=str(exception), original_exception=exception)    
    

    # List Models
    def list_models(self) -> list[str]:
        hugging_face_api = self.lazy_import('huggingface_hub.HfApi')()
        return hugging_face_api.list_models(library="sentence-transformers")
    

    # Embeddings
    def _get_embed_response(
            self,            
            input: list[str],
            model: str,
            dimensions: Optional[int] = None,
            **kwargs
            ) -> Any:
                      
        model_init_kwargs = {**self.client_kwargs, **kwargs.pop("model_init_kwargs", {})}
        truncate_dim = dimensions or model_init_kwargs.pop("truncate_dim", None)
        if not (st_model := self.st_model_cache.get(model)):
            # st_model = sentence_transformers.SentenceTransformer(
            st_model = self.lazy_import("sentence_transformers.SentenceTransformer")(
                model_name_or_path=model, 
                truncate_dim=truncate_dim,
                **model_init_kwargs
            )
            self.st_model_cache[model] = st_model        
        return st_model.encode(sentences=input, precision="float32", **kwargs)[:dimensions]
        

    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            **kwargs
            ) -> Embeddings:
        return Embeddings(root=response, response_info=ResponseInfo(model=model))
        

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

        model = model or self.default_reranking_model        
        model_init_kwargs = {**self.client_kwargs, **kwargs.pop("model_init_kwargs", {})}
        if not (ce_model := self.ce_model_cache.get(model)):
            # ce_model = sentence_transformers.CrossEncoder(
            ce_model = self.lazy_import("sentence_transformers.CrossEncoder")(
                model_name=model, 
                **model_init_kwargs
            )
            self.ce_model_cache[model] = ce_model     

        pairs = list(product([query], query_result.documents))   
        relevance_scores = ce_model.predict(pairs, **kwargs)
        return relevance_scores

    
    def _extract_reranked_order(
        self,
        response: Any,
        top_n: Optional[int] = None,        
        **kwargs
        ) -> list[int]:        
        return [index for index, score in sorted(enumerate(response), key=lambda x: x[1], reverse=True)[:top_n]]
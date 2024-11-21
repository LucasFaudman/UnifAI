from __future__ import annotations

from typing import Optional, Any, ClassVar, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from ...types import Embeddings, ResponseInfo
from ..base_adapters.sentence_transformers_base import SentenceTransformersAdapter
from ._base_embedder import Embedder
from ...utils import lazy_import


class SentenceTransformersEmbedder(SentenceTransformersAdapter, Embedder):
    provider = "sentence_transformers"
    default_embedding_model = "multi-qa-mpnet-base-cos-v1"

    # Cache for loaded SentenceTransformer models
    st_model_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    # Embeddings
    def _get_embed_response(
            self,            
            input: list[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Optional[Literal[
                "retrieval_query", 
                "retrieval_document", 
                "semantic_similarity", 
                "classification", 
                "clustering", 
                "question_answering", 
                "fact_verification", 
                "code_retrieval_query", 
                "image"]] = None,
            truncate: Literal[False, "end", "start"] = False,               
            **kwargs
            ) -> Any:
                      
        model_init_kwargs = {**self.client_kwargs, **kwargs.pop("model_init_kwargs", {})}
        truncate_dim = dimensions or model_init_kwargs.pop("truncate_dim", None)
        if not (st_model := self.st_model_cache.get(model)):
            # st_model = sentence_transformers.SentenceTransformer(
            st_model = lazy_import("sentence_transformers.SentenceTransformer")(
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
        

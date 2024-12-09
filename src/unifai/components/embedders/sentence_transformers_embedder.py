from __future__ import annotations

from typing import Optional, Any, ClassVar, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from ...types import Embeddings, ResponseInfo
from ..adapters.sentence_transformers_adapter import SentenceTransformersAdapter
from .._base_components._base_embedder import Embedder
from ...utils import lazy_import


class SentenceTransformersEmbedder(SentenceTransformersAdapter, Embedder):
    provider = "sentence_transformers"
    default_embedding_model = "multi-qa-mpnet-base-cos-v1"

    model_embedding_dimensions = {}

    # Cache for loaded SentenceTransformer models
    st_model_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def _get_model(self, model: str, dimensions: Optional[int] = None, **kwargs) -> SentenceTransformer:
        if not (st_model := self.st_model_cache.get(model)):
            model_init_kwargs = {**self.init_kwargs, **kwargs.pop("model_init_kwargs", {})}
            truncate_dim = dimensions or model_init_kwargs.pop("truncate_dim", None)

            st_model = lazy_import("sentence_transformers.SentenceTransformer")(
                model_name_or_path=model, 
                truncate_dim=truncate_dim,
                **model_init_kwargs
            )
            self.st_model_cache[model] = st_model
        return st_model

    def get_model_dimensions(self, model: Optional[str] = None) -> int:
        if model is None:
            model = self.default_model
        if self.config.extra_model_dimensions and (dimensions := self.config.extra_model_dimensions.get(model)):
            return dimensions
        if not (dimensions := self.model_embedding_dimensions.get(model)):
            st_model = self._get_model(model)
            if (dimensions := st_model.get_sentence_embedding_dimension()):
                self.model_embedding_dimensions[model] = dimensions
            else:
                dimensions = self.default_dimensions
        return dimensions

    
    def get_model_max_tokens(self, model: Optional[str] = None) -> int:
        if model is None:
            model = self.default_model
        if self.config.extra_model_max_tokens and (max_tokens := self.config.extra_model_max_tokens.get(model)):
            return max_tokens
        if not (max_tokens := self.model_max_tokens.get(model)):
            st_model = self._get_model(model)
            if (max_tokens := st_model.get_max_seq_length()):
                self.model_max_tokens[model] = max_tokens
            else:
                max_tokens = self.default_model_max_tokens
        return max_tokens

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
                      
        st_model = self._get_model(model, dimensions, **kwargs)     
        return st_model.encode(sentences=input, precision="float32", **kwargs)[:dimensions]
        

    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            **kwargs
            ) -> Embeddings:
        return Embeddings(root=response, response_info=ResponseInfo(model=model))
        

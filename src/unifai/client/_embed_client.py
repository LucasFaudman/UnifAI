from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.embedders._base_embedder import Embedder, Embeddings

from ._base_client import BaseClient


    
class UnifAIEmbedClient(BaseClient):

    def get_embedder(self, provider: Optional[str] = None, **client_kwargs) -> "Embedder":
        return self._get_component(provider, "embedder", **client_kwargs)

    # Embeddings
    def embed(
        self,
        input: str | Sequence[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
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
        input_too_large: Literal[
        "truncate_end", 
        "truncate_start", 
        "raise_error"] = "truncate_end",
        dimensions_too_large: Literal[
        "reduce_dimensions", 
        "raise_error"
        ] = "reduce_dimensions",
        task_type_not_supported: Literal[
        "use_closest_supported",
        "raise_error",
        ] = "use_closest_supported",                 
        **kwargs
                ) -> "Embeddings":
        
        return self.get_embedder(provider).embed(
            input, 
            model, 
            dimensions, 
            task_type, 
            input_too_large, 
            dimensions_too_large, 
            task_type_not_supported, 
            **kwargs
        )

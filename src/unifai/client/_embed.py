from typing import Optional, Sequence, Literal

from ..types import EmbeddingProvider, Embeddings
from .__client import get_embedder


def embed(
    input: str | Sequence[str],
    model: Optional[str] = None,
    provider: Optional[EmbeddingProvider] = None,
    dimensions: Optional[int] = None,
    task_type: Optional[Literal[
    "retreival_query", 
    "retreival_document", 
    "semantic_similarity", 
    "classification", 
    "clustering", 
    "question_answering", 
    "fact_verification", 
    "code_retreival_query", 
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
            ) -> Embeddings:
    
    return get_embedder(provider).embed(
        input, 
        model, 
        dimensions, 
        task_type, 
        input_too_large, 
        dimensions_too_large, 
        task_type_not_supported, 
        **kwargs
    )


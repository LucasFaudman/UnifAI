from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar
from ._base_configs import ComponentWithModelConfig

class EmbedderConfig(ComponentWithModelConfig):
    component_type: ClassVar = "embedder"
    default_dimensions: Optional[int] = None
    # default_task_type: Optional[Literal[
    #     "retrieval_document", 
    #     "retrieval_query", 
    #     "semantic_similarity", 
    #     "classification", 
    #     "clustering", 
    #     "question_answering", 
    #     "fact_verification", 
    #     "code_retrieval_query", 
    #     "image"]] = None
    # default_truncate: Literal[False, "end", "start"] = False
    # default_reduce_dimensions: bool = False
    # default_use_closest_supported_task_type: bool = True
    # default_batch_size: Optional[int] = None
    extra_model_dimensions: Optional[dict[str, int]] = None
    extra_model_max_tokens: Optional[dict[str, int]] = None
    extra_kwargs: Optional[dict[Literal["embed"], dict[str, Any]]] = None
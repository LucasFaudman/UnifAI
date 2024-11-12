from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import Embeddings, EmbeddingTaskTypeInput, ResponseInfo, Usage
from ...exceptions import ProviderUnsupportedFeatureError
from ..base_adapters.cohere_base import CohereAdapter
from ._base_embedder import Embedder

T = TypeVar("T")

class CohereEmbedder(CohereAdapter, Embedder):
    provider = "cohere"
    default_embedding_model = "embed-multilingual-v3.0"
    
    model_embedding_dimensions = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,        
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,      
    }


    # Embeddings
    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput] = None,
                            task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported"
                            ) -> Literal["search_document", "search_query", "classification", "clustering", "image"]:
        if task_type in ("classification", "clustering", "image"):
            return task_type
        elif task_type == "retrieval_query":
            return "search_query"        
        elif task_type == "retrieval_document" or task_type is None or task_type_not_supported == "use_closest_supported":
            return "search_document"     
        raise ProviderUnsupportedFeatureError(
             f"Embedding task_type={task_type} is not supported by Cohere. "
             "Supported input types are 'retrieval_query', 'retrieval_document', 'classification', 'clustering', 'image'")


    def _get_embed_response(
            self,            
            input: Sequence[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Literal["search_query", "search_document", "classification", "clustering", "image"] = "search_query",
            input_too_large: Literal[
                "truncate_end", 
                "truncate_start", 
                "raise_error"] = "truncate_end",
            **kwargs
            ) -> Any:
                
        if input_too_large == "truncate_end":
             truncate = "END"
        elif input_too_large == "truncate_start":
             truncate = "START"
        else:
             truncate = None # Raise error if input is too large
        
        return self.client.embed(
             model=model,
             **{"texts" if task_type != "image" else "images": input},             
             input_type=task_type,
             embedding_types=["float"],
             truncate=truncate,
             **kwargs
        )

             
    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            **kwargs
            ) -> Embeddings:        
        return Embeddings(
            root=response.embeddings.float,
            response_info=ResponseInfo(
                model=model, 
                usage=Usage(input_tokens=response.meta.billed_units.input_tokens)
            )
        )
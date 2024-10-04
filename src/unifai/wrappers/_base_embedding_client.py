from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator

from ._base_client_wrapper import BaseClientWrapper

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Usage, Embeddings, EmbeddingTaskTypeInput
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, EmbeddingDimensionsError

T = TypeVar("T")

class EmbeddingClient(BaseClientWrapper):
    provider = "base_embedding"
    default_embedding_model = "llama3.1:8b-instruct-q2_K"

    model_embedding_dimensions: dict[str, int] = {}

    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")    


    # Embeddings    
    def embed(
            self,            
            input: str | Sequence[str],
            model: Optional[str] = None,
            dimensions: Optional[int] = None,
            task_type: Optional[Literal[
                "retreival_document", 
                "retreival_query", 
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
        
        if input_too_large == "truncate_start" and self.provider not in ("nvidia", "cohere"):
            provider_title = self.provider.title()
            raise ProviderUnsupportedFeatureError(
                f"{provider_title} does not support truncating input at the start. "
                f"Use 'truncate_end' or 'raise_error' instead with {provider_title}. "
                "If you require truncating at the start, use Nvidia or Cohere embedding models which support this directly. "
                f"Or use 'raise_error' to handle truncation manually when the input is too large for {provider_title}."
                )                

        # Add to kwargs for passing to both getter (all) and extractor (needed by some ie Google)
        kwargs["input"] = [input] if isinstance(input, str) else input
        kwargs["model"] = (model := model or self.default_embedding_model)
        if dimensions is not None:
            dimensions = self.validate_dimensions(model, dimensions, dimensions_too_large)
        kwargs["dimensions"] = dimensions
        kwargs["task_type"] = self.validate_task_type(model, task_type, task_type_not_supported)
        kwargs["input_too_large"] = input_too_large
        # kwargs["dimensions_too_large"] = dimensions_too_large
        # kwargs["task_type_not_supported"] = task_type_not_supported

        response = self.run_func_convert_exceptions(
            func=self._get_embed_response,
            **kwargs
        )
        embeddings = self._extract_embeddings(response, **kwargs)
        if dimensions and dimensions < embeddings.dimensions:
            embeddings = embeddings.reduce_dimensions(dimensions)
        return embeddings

    def validate_dimensions(
            self, 
            model: str, 
            dimensions: int,
            dimensions_too_large: Literal["reduce_dimensions", "raise_error"] = "reduce_dimensions"                                      
            ) -> int:
        
        if dimensions is not None and dimensions < 1:
            raise EmbeddingDimensionsError(f"Embedding dimensions must be greater than 0. Got: {dimensions}")

        if ((model_dimensions := self.model_embedding_dimensions.get(model)) is None
            or dimensions <= model_dimensions):
            # Return as is if the model dimensions are unknown or smaller than the requested dimensions
            return dimensions
        
        if dimensions_too_large == "reduce_dimensions":
            # Reduce the dimensions to the model's maximum if the requested dimensions are too large
            return model_dimensions

        # Raise error if requested dimensions are too large for model before calling the API and wasting credits
        raise EmbeddingDimensionsError(
            f"Model {model} outputs at most {model_dimensions} dimensions, but {dimensions} were requested. Set dimensions_too_large='reduce_dimensions' to reduce the dimensions to {model_dimensions}"
        )


    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput],
                            task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported"
                            ) -> Any:
        if task_type and task_type_not_supported == "raise_error":
            provider_title = self.provider.title()
            raise ProviderUnsupportedFeatureError(
                f"Embedding Task Type {task_type} is not supported by {provider_title}. "
                f"If you require embeddings optimized for {task_type}, use Google or Cohere embedding models which support this directly. "
                f"Use 'use_closest_supported' to use the closest supported task type instead with {provider_title}. "
            )
        return task_type
        

    def _get_embed_response(
            self,            
            input: Sequence[str],
            model: str,
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
            ) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            **kwargs
            ) -> Embeddings:
        raise NotImplementedError("This method must be implemented by the subclass")


   
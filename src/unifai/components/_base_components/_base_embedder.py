from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Callable, Iterator, Iterable, Generator

from ._base_adapter import UnifAIAdapter

from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Usage, Embeddings, EmbeddingTaskTypeInput, Document, Documents
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError, EmbeddingDimensionsError
from ...utils import chunk_iterable
from ...configs.embedder_config import EmbedderConfig

T = TypeVar("T")
class Embedder(UnifAIAdapter[EmbedderConfig]):   
    component_type = "embedder"
    provider = "base"    
    config_class = EmbedderConfig
    
    model_embedding_dimensions: dict[str, int] = {}
    model_max_tokens: dict[str, int] = {}

    default_embedding_model = "llama3.1:8b-instruct-q2_K"
    default_embedding_dimensions = 768
    default_model_max_tokens = 2048

    # def __init__(self, config: Optional[EmbedderConfig] = None, **client_kwargs):
    #     super().__init__(config, **client_kwargs)

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
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            **kwargs
            ) -> Embeddings:
        raise NotImplementedError("This method must be implemented by the subclass")    

    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    @property
    def default_model(self) -> str:
        return self.config.default_model or self.default_embedding_model
        
    @property
    def default_dimensions(self) -> int:
        return self.config.default_dimensions or self.default_embedding_dimensions
    
    def get_model_dimensions(self, model: Optional[str] = None) -> int:
        if model is None:
            model = self.default_model
        if self.config.extra_model_dimensions and (dimensions := self.config.extra_model_dimensions.get(model)):
            return dimensions
        return self.model_embedding_dimensions.get(model) or self.default_dimensions 
    
    def get_model_max_tokens(self, model: Optional[str] = None) -> int:
        if model is None:
            model = self.default_model
        if self.config.extra_model_max_tokens and (max_tokens := self.config.extra_model_max_tokens.get(model)):
            return max_tokens
        return self.model_max_tokens.get(model) or self.default_model_max_tokens
    
    def validate_dimensions(
            self, 
            model: str, 
            dimensions: Optional[int],
            reduce_dimensions: bool                                    
            ) -> Optional[int]:
        
        if dimensions is None:
            return self.get_model_dimensions(model)
        if dimensions < 1:
            raise EmbeddingDimensionsError(f"Embedding dimensions must be greater than 0. Got: {dimensions}")
                        
        # Return as is if the model dimensions are unknown or smaller than the requested dimensions
        if dimensions <= (model_dimensions := self.get_model_dimensions(model)):
            return dimensions
        # Reduce the dimensions to the model's maximum if the requested dimensions are too large                
        if reduce_dimensions:
            return model_dimensions
        # Raise error if requested dimensions are too large for model before calling the API and wasting credits
        raise EmbeddingDimensionsError(
            f"Model {model} outputs at most {model_dimensions} dimensions, but {dimensions} were requested. Set reduce_dimensions=True to reduce the dimensions to {model_dimensions}"
        )


    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput],
                            use_closest_supported_task_type: bool
                            ) -> Optional[EmbeddingTaskTypeInput]:
        # Default is to only allow task_type=None unless overridden by subclasses
        if task_type and not use_closest_supported_task_type:
            provider_title = self.provider.title()
            raise ProviderUnsupportedFeatureError(
                f"Embedding Task Type {task_type} is not supported by {provider_title}. "
                f"If you require embeddings optimized for {task_type}, use Google or Cohere embedding models which support this directly. "
                f"Use 'use_closest_supported' to use the closest supported task type instead with {provider_title}. "
            )
        return task_type
        
    def validate_truncate(
            self,
            model: str,
            truncate: Literal[False, "end", "start"]
            ) -> Literal[False, "end", "start"]:
                
        # Default is to only allow start if provider is Nvidia or Cohere. Can be overridden by subclasses
        if truncate == "start" and self.provider not in ("nvidia", "cohere"):
            provider_title = self.provider.title()
            raise ProviderUnsupportedFeatureError(
                f"{provider_title} does not support truncating input at the start. "
                f"Use 'truncate_end' or 'raise_error' instead with {provider_title}. "
                "If you require truncating at the start, use Nvidia or Cohere embedding models which support this directly. "
                f"Or use 'raise_error' to handle truncation manually when the input is too large for {provider_title} {model}."
                )         
        return truncate    

    # Embeddings    
    def embed(
            self,            
            input: str | list[str],
            model: Optional[str] = None,
            dimensions: Optional[int] = None,
            task_type: Optional[Literal[
                "retrieval_document", 
                "retrieval_query", 
                "semantic_similarity", 
                "classification", 
                "clustering", 
                "question_answering", 
                "fact_verification", 
                "code_retrieval_query", 
                "image"]] = None,
            truncate: Literal[False, "end", "start"] = False,
            reduce_dimensions: bool = False,
            use_closest_supported_task_type: bool = True,        
            **kwargs
            ) -> Embeddings:
        
        # Add to kwargs for passing to both getter (all) and extractor (needed by some ie Google, some Nvidia models)
        kwargs["input"] = [input] if isinstance(input, str) else input
        kwargs["model"] = (model := model or self.default_model)
        # Validate and set dimensions. Raises error if dimensions are invalid or too large for the model
        kwargs["dimensions"] = self.validate_dimensions(model, dimensions, reduce_dimensions)
        # Validate and set task type. Raises error if task type is not supported by the provider
        kwargs["task_type"] = self.validate_task_type(model, task_type, use_closest_supported_task_type)
        # Validate and set truncate. Raises error if truncation is not supported by the provider
        kwargs["truncate"] = self.validate_truncate(model, truncate)

        if self.config.extra_kwargs and (extra_kwargs := self.config.extra_kwargs.get("embed")):
            kwargs.update(extra_kwargs)

        response = self.run_func_convert_exceptions(
            func=self._get_embed_response,
            **kwargs
        )
        embeddings = self._extract_embeddings(response, **kwargs)
        if dimensions and dimensions < embeddings.dimensions:
            embeddings.reduce_dimensions(dimensions)
        return embeddings

    def iembed_documents(
            self,            
            documents: Iterable[Document],
            model: Optional[str] = None,
            dimensions: Optional[int] = None,
            task_type: Optional[Literal[
                "retrieval_document", 
                "retrieval_query", 
                "semantic_similarity", 
                "classification", 
                "clustering", 
                "question_answering", 
                "fact_verification", 
                "code_retrieval_query", 
                "image"]] = None,
            truncate: Literal[False, "end", "start"] = False,
            reduce_dimensions: bool = False,
            use_closest_supported_task_type: bool = True,      
            batch_size: Optional[int] = None,           
            **kwargs
            ) -> Generator[Document, None, ResponseInfo]:
        
        response_info = ResponseInfo()
        if batch_size is not None:
            for batch in chunk_iterable(documents, batch_size):
                texts = [document.text or "" for document in batch]
                embeddings = self.embed(texts, model, dimensions, task_type, truncate, reduce_dimensions, use_closest_supported_task_type, **kwargs)
                response_info += embeddings.response_info
                for document, embedding in zip(batch, embeddings):
                    document.embedding = embedding
                    yield document
        else:
            for document in documents:
                text = document.text or ""
                embeddings = self.embed(text, model, dimensions, task_type, truncate, reduce_dimensions, use_closest_supported_task_type, **kwargs)
                response_info += embeddings.response_info
                document.embedding = embeddings[0]
                yield document
        return response_info
    
    def embed_documents(
            self,            
            documents: Iterable[Document],
            model: Optional[str] = None,
            dimensions: Optional[int] = None,
            task_type: Optional[Literal[
                "retrieval_document", 
                "retrieval_query", 
                "semantic_similarity", 
                "classification", 
                "clustering", 
                "question_answering", 
                "fact_verification", 
                "code_retrieval_query", 
                "image"]] = None,
            truncate: Literal[False, "end", "start"] = False,
            reduce_dimensions: bool = False,
            use_closest_supported_task_type: bool = True,  
            batch_size: Optional[int] = None,           
            **kwargs
            ) -> Documents:
        return Documents.from_generator(self.iembed_documents(
            documents, model, dimensions, task_type, truncate, reduce_dimensions, use_closest_supported_task_type, batch_size, **kwargs
        ))



from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_client_wrapper import BaseClientWrapper, UnifAIExceptionConverter
from ._base_embedding_client import EmbeddingClient, EmbeddingTaskTypeInput
from ._base_llm_client import LLMClient
from ._base_reranker_client import RerankerClient

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, EmbeddingDimensionsError

from cohere import ClientV2
from cohere.core import ApiError as CohereAPIError

T = TypeVar("T")

class CohereWrapper(EmbeddingClient, RerankerClient, LLMClient):
    client: ClientV2

    provider = "cohere"
    default_model = "command"
    default_embedding_model = "embed-multilingual-v3.0"
    default_reranking_model = "rerank-multilingual-v3.0"
    
    model_embedding_dimensions = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,        
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,      
    }

    def import_client(self):
        from cohere import ClientV2
        return ClientV2
    
    
    def init_client(self, **client_kwargs) -> ClientV2:
        self.client_kwargs.update(client_kwargs)
        if not (api_key := self.client_kwargs.get("api_key")):
            raise ValueError("Cohere API key is required")
        self._client = self.import_client()(api_key) # Cohere ClientV2 does require an API key as a positional argument
        return self._client


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: CohereAPIError) -> UnifAIError:
        message = exception.body
        status_code = exception.status_code
        if status_code is not None:
                unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
                unifai_exception_type = UnknownAPIError
        return unifai_exception_type(
            message=message,
            status_code=status_code,
            original_exception=exception
        )        
    

    # Embeddings
    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput] = None,
                            task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported"
                            ) -> Literal["search_document", "search_query", "classification", "clustering", "image"]:
        if task_type in ("classification", "clustering", "image"):
            return task_type
        elif task_type == "retreival_query":
            return "search_query"        
        elif task_type == "retreival_document" or task_type is None or task_type_not_supported == "use_closest_supported":
            return "search_document"     
        raise ProviderUnsupportedFeatureError(
             f"Embedding task_type={task_type} is not supported by Cohere. "
             "Supported input types are 'retreival_query', 'retreival_document', 'classification', 'clustering', 'image'")


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

        return self.client.rerank(
             model=model,
             query=query,
             documents=query_result.documents,
             top_n=top_n,
             **kwargs
        )
        

    def _extract_reranked_order(
        self,
        response: Any,
        top_n: Optional[int] = None,        
        **kwargs
        ) -> list[int]:
        sorted_results = sorted(response.results, key=lambda result: result.relevance_score, reverse=True)
        if top_n is not None and top_n < len(sorted_results):
            sorted_results = sorted_results[:top_n]
        return [result.index for result in sorted_results]
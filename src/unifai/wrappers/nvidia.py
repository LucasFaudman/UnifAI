from typing import Optional, Union, Any, Literal, Mapping, Iterator, Sequence, Generator
from json import loads as json_loads, JSONDecodeError
from datetime import datetime

from openai import OpenAI
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDeltaToolCall, Choice as ChunkChoice
from openai.types.chat.chat_completion import ChatCompletion, Choice as CompletionChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai import (
    OpenAIError,
    APIError as OpenAIAPIError,
    APIConnectionError as OpenAIAPIConnectionError,
    APITimeoutError as OpenAIAPITimeoutError,
    APIResponseValidationError as OpenAIAPIResponseValidationError,
    APIStatusError as OpenAIAPIStatusError,
    AuthenticationError as OpenAIAuthenticationError,
    BadRequestError as OpenAIBadRequestError,
    ConflictError as OpenAIConflictError,
    InternalServerError as OpenAIInternalServerError,
    NotFoundError as OpenAINotFoundError,
    PermissionDeniedError as OpenAIPermissionDeniedError,
    RateLimitError as OpenAIRateLimitError,
    UnprocessableEntityError as OpenAIUnprocessableEntityError,
)

from unifai.exceptions import (
    UnifAIError,
    APIError,
    UnknownAPIError,
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    STATUS_CODE_TO_EXCEPTION_MAP,
    ProviderUnsupportedFeatureError
)

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, Usage, ResponseInfo, Embeddings, EmbeddingTaskTypeInput
from unifai.type_conversions import stringify_content
from ._base_reranker_client import RerankerClient
from .openai import OpenAIWrapper 

class NvidiaWrapper(OpenAIWrapper, RerankerClient):
    provider = "nvidia"
    default_model = "meta/llama-3.1-405b-instruct"
    
    # Nvidia API is OpenAI Compatible 
    # (with minor differences: 
    # - available models
    # - image input format
    #   - (Nvidia uses HTML <img src=\"data:image/png;base64,iVBORw .../> 
    #      while OpenAI uses data_uri/url {'type':'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw ...'}})
    # - embedding parameters (truncate, input_type, etc)
    # - with many more with tbd)
    default_base_url = "https://integrate.api.nvidia.com/v1"
    retreival_base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    
    default_embedding_model = "nvidia/nv-embed-v1" #NV-Embed-QA
    default_multimodal_model = "NV-Embed-QA" 

    model_embedding_dimensions = {
        "baai/bge-m3": 1024,
        "NV-Embed-QA": 1024,
        "nvidia/nvclip": 1024,
        "nvidia/nv-embed-v1": 4096,
        "nvidia/nv-embedqa-e5-v5": 1024,
        "nvidia/nv-embedqa-mistral-7b-v2": 4096,
        "snowflake/arctic-embed-l": 1024,
    }    

    model_base_urls = {
        "NV-Embed-QA": retreival_base_url,
        "nvidia/nv-rerankqa-mistral-4b-v3": retreival_base_url,
        "snowflake/arctic-embed-l": retreival_base_url+"snowflake/arctic-embed-l/",
    } 

    default_reranking_model = "nv-rerank-qa-mistral-4b:1"



    def init_client(self, **client_kwargs) -> Any:
        if "base_url" not in client_kwargs:
            # Add the Nvidia base URL if not provided since the default is OpenAI
            client_kwargs["base_url"] = self.default_base_url
        return super().init_client(**client_kwargs)
  

    # Embeddings (Only override OpenAIWrapper if necessary)
    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput] = None,
                            task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported"
                            ) -> Literal["query", "passage"]:

        if task_type == "retreival_query":
            return "query"        
        elif task_type == "retreival_document" or task_type is None or task_type_not_supported == "use_closest_supported":
            return "passage"     
        raise ProviderUnsupportedFeatureError(
             f"Embedding task_type={task_type} is not supported by Nvidia. "
             "Supported input types are 'retreival_query', 'retreival_document'")
    
        
    def _get_embed_response(
            self,
            input: Sequence[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Literal["passage", "query"] = "passage",
            input_too_large: Literal[
                "truncate_end", 
                "truncate_start", 
                "raise_error"] = "truncate_end",
            **kwargs
            ) -> CreateEmbeddingResponse:
        
        extra_body = {"input_type": task_type}
        if input_too_large == "truncate_end":
            extra_body["truncate"] = "END"
        elif input_too_large == "truncate_start":
            extra_body["truncate"] = "START"
        else:
            extra_body["truncate"] = "NONE" # Raise error if input is too large
        
        # model = f"{model}-{task_type}" 
        return self.client.embeddings.create(input=input, model=model, extra_body=extra_body, **kwargs)  

    

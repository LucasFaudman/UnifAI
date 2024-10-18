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
from openai._utils import maybe_transform
from openai._base_client import make_request_options
from openai.types.chat.completion_create_params import CompletionCreateParams

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, Usage, ResponseInfo, Embeddings, EmbeddingTaskTypeInput, VectorDBQueryResult
from unifai.type_conversions import stringify_content
from ..components.rerankers._base_reranker import Reranker
from .openai import OpenAIAdapter


from typing_extensions import Literal, Required, TypeAlias, TypedDict


class TempBaseURL:
    """
    Temporarily change the base URL of the client to the provided base URL and then reset it after exiting the context

    Nvidia API requires different base URLs for different models unlike OpenAI which uses the same base URL for all models and endpoints

    Args:
        client (OpenAI): The OpenAI client
        base_url (Optional[str]): The new base URL to use
        default_base_url (str): The default base URL to reset to after exiting the context 
    """

    def __init__(self, 
                 client: OpenAI, 
                 base_url: Optional[str], 
                 default_base_url: str
                 ):
        self.client = client
        self.base_url = base_url
        self.default_base_url = default_base_url

    def __enter__(self):
        if self.base_url:
            self.client.base_url = self.base_url

    def __exit__(self, exc_type, exc_value, traceback):
        if self.base_url:
            self.client.base_url = self.default_base_url


class NvidiaAdapter(OpenAIAdapter, Reranker):
    provider = "nvidia"
    default_model = "meta/llama-3.1-405b-instruct"
    
    # Nvidia API is (kinda) OpenAI Compatible 
    # (with minor differences: 
    # - available models
    # - base URLs for vary for different models (sometimes) or tasks (usually)
    # - image input format
    #   - (Nvidia uses HTML <img src=\"data:image/png;base64,iVBORw .../> 
    #      while OpenAI uses data_uri/url {'type':'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw ...'}})
    # - embedding parameters (truncate, input_type, etc)
    # - probably many with time)
    default_base_url = "https://integrate.api.nvidia.com/v1"
    retreival_base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia"
    vlm_base_url = "https://ai.api.nvidia.com/v1/vlm/"
    
    default_embedding_model = "nvidia/nv-embed-v1" #NV-Embed-QA
    default_reranking_model = "nvidia/nv-rerankqa-mistral-4b-v3"
    default_multimodal_model = "nvidia/vila" 

    model_embedding_dimensions = {
        "baai/bge-m3": 1024,
        "NV-Embed-QA": 1024,
        "nvidia/nvclip": 1024,
        "nvidia/nv-embed-v1": 4096,
        "nvidia/nv-embedqa-e5-v5": 1024,
        "nvidia/nv-embedqa-mistral-7b-v2": 4096,
        "snowflake/arctic-embed-l": 1024,
    }    

    reranking_models = {
        "nvidia/nv-rerankqa-mistral-4b-v3",
        "nv-rerank-qa-mistral-4b:1"        
    }

    vlm_models = {
        "adept/fuyu-8b",
        "google/deplot",
        "google/paligemma",
        # "liuhaotian/llava-v1.6-34b",
        # "liuhaotian/llava-v1.6-mistral-7b",
        # "community/llava-v1.6-34b",
        # "community/llava-v1.6-mistral-7b",        
        # "meta/llama-3.2-11b-vision-instruct",
        # "meta/llama-3.2-90b-vision-instruct",
        # "microsoft/florence-2"
        "microsoft/kosmos-2",
        "microsoft/phi-3-vision-128k-instruct",
        "nvidia/neva-22b",
        "nvidia/vila"
    }

    model_base_urls = {
        "NV-Embed-QA": retreival_base_url,
        "nv-rerank-qa-mistral-4b:1": retreival_base_url,
        "nvidia/nv-rerankqa-mistral-4b-v3": f"{retreival_base_url}/nv-rerankqa-mistral-4b-v3",
        "snowflake/arctic-embed-l": f"{retreival_base_url}/snowflake/arctic-embed-l",
        # "meta/llama-3.2-11b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
        # "meta/llama-3.2-90b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
    } 

    _openai_compatible_nvidia_rerank_response = None

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
        
        # Use the model specific base URL if required
        if other_base_url := self.model_base_urls.get(model):
            self.client.base_url = other_base_url
        
        respone = self.client.embeddings.create(input=input, model=model, extra_body=extra_body, **kwargs)
        
        # Reset the base URL if changed
        if other_base_url:
            self.client.base_url = self.default_base_url
        
        return respone


    # Reranking
    def _get_openai_compatible_nvidia_rerank_response(self) -> type:
        """
        Entire point of this is to have a castable type subclassing OpenAI's BaseModel so it does not
        raise TypeError("Pydantic models must subclass our base model type, e.g. `from openai import BaseModel`")
        and ensure it is only created once and at runtime (not recreated every call and only created if needed)
        """
        if self._openai_compatible_nvidia_rerank_response is None:
            from openai import BaseModel 
            class NvidiaRerankItem(BaseModel):
                index: int
                logit: float

            class NvidiaRerankResponse(BaseModel):
                rankings: list[NvidiaRerankItem]

            self._openai_compatible_nvidia_rerank_response = NvidiaRerankResponse
        return self._openai_compatible_nvidia_rerank_response


    def _get_rerank_response(
        self,
        query: str,
        query_result: VectorDBQueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:

        assert query_result.documents, "No documents to rerank"
        body = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": document} for document in query_result.documents],
        }

        options = {}
        if (
            (extra_headers := kwargs.get("extra_headers"))
            or (extra_query := kwargs.get("extra_query"))
            or (extra_body := kwargs.get("extra_body"))
            or (timeout := kwargs.get("timeout"))
        ):
            options["options"] = make_request_options(
                extra_headers=extra_headers, 
                extra_query=extra_query, 
                extra_body=extra_body, 
                timeout=timeout
            )

        # Use the reranking model specific base URL (always required)
        model_base_url = self.model_base_urls.get(model)
        with TempBaseURL(self.client, model_base_url, self.default_base_url):
            return self.client.post(
                "/reranking",
                body=body,
                **options,
                cast_to=self._get_openai_compatible_nvidia_rerank_response(), # See above
                stream=False,
                stream_cls=None,
            )


    def _extract_reranked_order(
        self,
        response: Any,
        top_n: Optional[int] = None,
        **kwargs
        ) -> list[int]:
        return [item.index for item in response.rankings]
    

    # Chat
    def _create_completion(self, kwargs) -> ChatCompletion|Stream[ChatCompletionChunk]:        
        if kwargs.get("model") not in self.vlm_models:
            # For non-multimodal models, use the default OpenAI completion method
            return self.client.chat.completions.create(**kwargs)
                
        model = kwargs.get("model")
        stream = kwargs.get("stream", None)
        extra_headers = kwargs.pop("extra_headers", None)
        extra_query = kwargs.pop("extra_query", None)
        extra_body = kwargs.pop("extra_body", None)
        timeout = kwargs.pop("timeout", None)

        # Get VLM base URL for the model or use the default VLM base URL
        model_base_url = self.model_base_urls.get(model, self.vlm_base_url)
        with TempBaseURL(self.client, model_base_url, self.default_base_url):
            return self.client.post(
                f"/{model}",
                body=maybe_transform(
                    kwargs,
                    CompletionCreateParams,
                ),
                options=make_request_options(
                    extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
                ),
                cast_to=ChatCompletion,
                stream=stream or False,
                stream_cls=Stream[ChatCompletionChunk],
            )

    # Convert from UnifAI to AI Provider format        
        # Messages     
    def format_user_message(self, message: Message) -> dict:
        message_dict = {"role": "user"}
        content = message.content
             
        if message.images:
            if not content:
                content = ""
            if content:
                content += " "
            content += " ".join(map(self.format_image, message.images))
                    
        message_dict["content"] = content
        return message_dict    
    
        # Images
    def format_image(self, image: Image) -> str:
        return f"<img src=\"{image.data_uri}\" />"

        # Tools
    def format_tool(self, tool: Tool) -> dict:
        return tool.to_dict(exclude=("strict", "additionalProperties"))
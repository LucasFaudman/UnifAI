from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion

from openai._utils import maybe_transform
from openai._base_client import make_request_options
from openai.types.chat.completion_create_params import CompletionCreateParams

from unifai.types import Message, Image, Tool
from ..base_adapters.nvidia_base import NvidiaAdapter, TempBaseURL
from .openai_llm import OpenAILLM


class NvidiaLLM(NvidiaAdapter, OpenAILLM):
    provider = "nvidia"
    default_model = "meta/llama-3.1-405b-instruct"
    
    vlm_base_url = "https://ai.api.nvidia.com/v1/vlm/"
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
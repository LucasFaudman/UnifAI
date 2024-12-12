from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from openai import OpenAI

from .openai_adapter import OpenAIAdapter

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


class NvidiaAdapter(OpenAIAdapter):
    provider = "nvidia"
    
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
    retrieval_base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia"
    vlm_base_url = "https://ai.api.nvidia.com/v1/vlm/"
  
    model_base_urls = {
        "NV-Embed-QA": retrieval_base_url,
        "nv-rerank-qa-mistral-4b:1": retrieval_base_url,
        "nvidia/nv-rerankqa-mistral-4b-v3": f"{retrieval_base_url}/nv-rerankqa-mistral-4b-v3",
        "snowflake/arctic-embed-l": f"{retrieval_base_url}/snowflake/arctic-embed-l",
        # "meta/llama-3.2-11b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
        # "meta/llama-3.2-90b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
    } 

    def init_client(self, **init_kwargs):
        if "base_url" not in init_kwargs:
            # Add the Nvidia base URL if not provided since the default is OpenAI
            init_kwargs["base_url"] = self.default_base_url
        return super().init_client(**init_kwargs)
  
    def _list_models(self) -> list[str]:
        return super()._list_models()
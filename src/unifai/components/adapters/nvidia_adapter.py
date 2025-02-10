from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from openai import OpenAI

from .openai_adapter import OpenAIAdapter, TempBaseURL

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

  
    def _list_models(self) -> list[str]:
        return super()._list_models()
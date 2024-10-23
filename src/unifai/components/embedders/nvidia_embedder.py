from typing import Optional, Union, Any, Literal, Mapping, Iterator, Sequence, Generator

from openai.types.create_embedding_response import CreateEmbeddingResponse

from ...exceptions import ProviderUnsupportedFeatureError
from ...types import EmbeddingTaskTypeInput
from ..base_adapters.nvidia_base import NvidiaAdapter, TempBaseURL
from .openai_embedder import OpenAIEmbedder


class NvidiaEmbedder(NvidiaAdapter, OpenAIEmbedder):
    provider = "nvidia"    
    default_embedding_model = "nvidia/nv-embed-v1" #NV-Embed-QA

    model_embedding_dimensions = {
        "baai/bge-m3": 1024,
        "NV-Embed-QA": 1024,
        "nvidia/nvclip": 1024,
        "nvidia/nv-embed-v1": 4096,
        "nvidia/nv-embedqa-e5-v5": 1024,
        "nvidia/nv-embedqa-mistral-7b-v2": 4096,
        "snowflake/arctic-embed-l": 1024,
    }    


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
        with TempBaseURL(self.client, self.model_base_urls.get(model), self.default_base_url):
            return self.client.embeddings.create(input=input, model=model, extra_body=extra_body, **kwargs)

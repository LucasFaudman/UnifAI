from typing import Optional, Union, Any, Literal, Mapping, Iterator, Sequence, Generator

from openai.types.create_embedding_response import CreateEmbeddingResponse

from ...exceptions import ProviderUnsupportedFeatureError
from ...types import EmbeddingTaskTypeInput
from ..adapters.nvidia_adapter import NvidiaAdapter, TempBaseURL
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
                            use_closest_supported_task_type: bool = True,        
                            ) -> Literal["query", "passage"]:
        
        if task_type == "retrieval_query":
            return "query"        
        elif task_type in ("retrieval_document", None) or use_closest_supported_task_type:
            return "passage"     
        raise ProviderUnsupportedFeatureError(
             f"Embedding task_type={task_type} is not supported by Nvidia. "
             "Supported input types are 'retrieval_query', 'retrieval_document'")
    
        
    def _get_embed_response(
            self,
            input: list[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Literal["passage", "query"] = "passage",
            truncate: Literal[False, "end", "start"] = False,
            **kwargs
            ) -> CreateEmbeddingResponse:
        
        extra_body = {"input_type": task_type}
        extra_body["truncate"] = truncate.upper() if truncate else "NONE"
        # Use the model specific base URL if required
        with TempBaseURL(self.client, self.model_base_urls.get(model), self.default_base_url):
            return self.client.embeddings.create(input=input, model=model, extra_body=extra_body, **kwargs)

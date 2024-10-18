from typing import Optional, Union, Sequence, Any, Literal, Mapping,  Iterator, Iterable, Generator, Collection

from ollama._types import Options as OllamaOptions
from ...types import Embeddings, ResponseInfo, Usage
from ..base_adapters.ollama_base import OllamaAdapter
from ._base_embedder import Embedder


class OllamaEmbedder(OllamaAdapter, Embedder):
    provider = "ollama"
    default_embedding_model = "llama3.1:8b-instruct-q2_K"

    # Embeddings
    def _get_embed_response(
            self,            
            input: Sequence[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Optional[str] = None,
            input_too_large: Literal[
                "truncate_end", 
                "truncate_start", 
                "raise_error"
                ] = "truncate_end",
            **kwargs
            )-> Mapping:
        return self.client.embed(
            input=input,
            model=model,
            truncate=(input_too_large != "raise_error"),
            keep_alive=kwargs.pop('keep_alive', None),
            options=OllamaOptions(**kwargs) if kwargs else None
        )


    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            **kwargs
            ) -> Embeddings:
        return Embeddings(
            root=response["embeddings"],
            response_info=ResponseInfo(
                model=model, 
                usage=Usage(
                    input_tokens=response['prompt_eval_count'], 
                )
            )
        )     
  

from typing import Optional, Union, Any, Literal, Mapping, Iterator, Sequence, Generator

from openai.types.create_embedding_response import CreateEmbeddingResponse

from ...types import Embeddings, ResponseInfo, Usage
from ..base_adapters.openai_base import OpenAIAdapter
from ._base_embedder import Embedder

class OpenAIEmbedder(OpenAIAdapter, Embedder):
    provider = "openai"
    default_embedding_model = "text-embedding-3-large"
    model_embedding_dimensions = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }

    # Embeddings
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
            ) -> CreateEmbeddingResponse:
        
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return self.client.embeddings.create(input=input, model=model, **kwargs)


    def _extract_embeddings(
            self,            
            response: CreateEmbeddingResponse,
            **kwargs
            ) -> Embeddings:
        return Embeddings(
            root=[embedding.embedding for embedding in response.data],
            response_info=ResponseInfo(
                model=response.model, 
                usage=Usage(
                    input_tokens=response.usage.prompt_tokens, 
                    output_tokens=0
                )
            )
        )

 
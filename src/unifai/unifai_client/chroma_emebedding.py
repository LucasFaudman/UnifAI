import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)
from typing import Optional
from unifai.types.response_info import ResponseInfo
from pathlib import Path

def get_chroma_client(path: str|Path):
    return chromadb.PersistentClient(
    path=path if isinstance(path, str) else str(path),
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
    )

class UnifAIChromaEmbeddingFunction(EmbeddingFunction[list[str]]):
    def __init__(
            self,
            parent,
            provider: Optional[str] = None,
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            response_infos: Optional[list[ResponseInfo]] = None,
    ):
        self.parent = parent
        self.provider = provider or parent.default_provider
        self.model = model
        self.max_dimensions = max_dimensions
        self.response_infos = response_infos or []

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        embded_result = self.parent.embed(
            input=input,
            model=self.model,
            provider=self.provider,
            max_dimensions=self.max_dimensions
        )
        self.response_infos.append(embded_result.response_info)
        return [embedding.vector for embedding in embded_result]
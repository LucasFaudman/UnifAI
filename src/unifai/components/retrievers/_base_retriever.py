from typing import Optional, Literal

from ...types import Embedding, VectorDBQueryResult
from .._base_component import UnifAIComponent

class Retriever(UnifAIComponent):
    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:      
        
        raise NotImplementedError("This method must be implemented by the subclass")
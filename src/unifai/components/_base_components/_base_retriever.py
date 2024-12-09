from typing import Optional, Literal

from ...types import Embedding, QueryResult, RankedDocument, Embeddings
from ._base_component import UnifAIComponent

class Retriever(UnifAIComponent):
    component_type = "retriever"
    provider = "base"    
      
    def query(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> QueryResult:        
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def query_documents(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> list[RankedDocument]:        
        return self.query(query_input, top_k, where, where_document, include, **kwargs).to_documents()   
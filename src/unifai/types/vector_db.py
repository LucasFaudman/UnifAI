from typing import Optional, Sequence, Literal, Self
from pydantic import BaseModel, RootModel

from .embeddings import Embeddings, Embedding

class VectorDBGetResult(BaseModel):
    ids: list[str]
    embeddings: Optional[list[Embedding]]
    documents: Optional[list[str]]
    metadatas: Optional[list[dict]]    
    included: Sequence[Literal["embeddings", "metadatas", "documents"]]


class VectorDBQueryResult(VectorDBGetResult):
    distances: Optional[list[float]]
    included: Sequence[Literal["embeddings", "metadatas", "documents", "distances"]]


    def rerank(self, reranked_order: Sequence[int]) -> Self:
        old_ids = self.ids.copy()
        old_embeddings = self.embeddings.copy() if self.embeddings else None
        old_documents = self.documents.copy() if self.documents else None
        old_metadatas = self.metadatas.copy() if self.metadatas else None
        old_distances = self.distances.copy() if self.distances else None
        for i in reranked_order:
            self.ids[i] = old_ids[i]
            if self.embeddings and old_embeddings:
                self.embeddings[i] = old_embeddings[i]
            if self.documents and old_documents:
                self.documents[i] = old_documents[i]
            if self.metadatas and old_metadatas:
                self.metadatas[i] = old_metadatas[i]
            if self.distances and old_distances:
                self.distances[i] = old_distances[i]
        return self
    

    def reduce_to_top_n(self, n: int) -> Self:
        if n >= len(self.ids):
            return self

        self.ids = self.ids[:n]
        if self.embeddings:
            self.embeddings = self.embeddings[:n]
        if self.documents:
            self.documents = self.documents[:n]
        if self.metadatas:
            self.metadatas = self.metadatas[:n]
        if self.distances:
            self.distances = self.distances[:n]
        return self


    def slice(self, start: int, end: int) -> Self:
        self.ids = self.ids[start:end]
        if self.embeddings:
            self.embeddings = self.embeddings[start:end]
        if self.documents:
            self.documents = self.documents[start:end]
        if self.metadatas:
            self.metadatas = self.metadatas[start:end]
        if self.distances:
            self.distances = self.distances[start:end]
        return self
    

    def __iter__(self):
        return iter(self.ids)
    
    def __len__(self):
        return len(self.ids)


class VectorDBQueryResults(RootModel[list[VectorDBQueryResult]]):
    pass
    # results: list[VectorDBQueryResult]
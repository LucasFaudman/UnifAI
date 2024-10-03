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
        for old_index, new_index in enumerate(reranked_order):
            self.ids[new_index] = old_ids[old_index]
            if self.embeddings and old_embeddings:
                self.embeddings[new_index] = old_embeddings[old_index]
            if self.documents and old_documents:
                self.documents[new_index] = old_documents[old_index]
            if self.metadatas and old_metadatas:
                self.metadatas[new_index] = old_metadatas[old_index]
            if self.distances and old_distances:
                self.distances[new_index] = old_distances[old_index]
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
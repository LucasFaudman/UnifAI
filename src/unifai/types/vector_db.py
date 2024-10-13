from typing import Optional, Sequence, Literal, Self, Callable
from pydantic import BaseModel, RootModel

from .embeddings import Embeddings, Embedding

class VectorDBGetResult(BaseModel):
    ids: list[str]
    embeddings: Optional[list[Embedding]]
    documents: Optional[list[str]]
    metadatas: Optional[list[dict]]    
    included: list[Literal["ids", "embeddings", "metadatas", "documents"]]


    def __iter__(self):
        return iter(self.ids)
    

    def __len__(self):
        return len(self.ids)
    
    
    def rerank(self, new_order: Sequence[int]) -> Self:
        old = {attr: value.copy() for attr in self.included if (value := getattr(self, attr)) is not None}
        for attr in self.included:
            new = getattr(self, attr)
            for new_index, old_index in enumerate(new_order):
                new[new_index] = old[attr][old_index]
        return self
    

    def slice(self, start: Optional[int] = None, end: Optional[int] = None) -> Self:
        for attr in self.included:
            setattr(self, attr, getattr(self, attr)[start:end])
        return self


    def reduce_to_top_n(self, n: int) -> Self:
        return self.slice(end=n)
    

    def sort(
            self,
            by: Literal["ids", "embeddings", "documents", "metadatas", "distances"] = "ids",
            key: Optional[Callable] = None,
            reverse: bool = False
    ) -> Self:
        
        _key = (lambda x: key(x[1])) if key else (lambda x: x[1])
        new_order = [x[0] for x in sorted(enumerate(getattr(self, by)), key=_key, reverse=reverse)]
        return self.rerank(new_order)
        

class VectorDBQueryResult(VectorDBGetResult):
    distances: Optional[list[float]]
    included: list[Literal["ids", "embeddings", "metadatas", "documents", "distances"]]


from typing import Optional, Sequence, Literal, Self, Callable, Iterator
from pydantic import BaseModel, RootModel

from .documents import Document
from .embeddings import Embeddings, Embedding
from itertools import zip_longest

class GetResult(BaseModel):
    ids: list[str]
    embeddings: Optional[list[Embedding]]
    documents: Optional[list[str]]
    metadatas: Optional[list[dict]]    
    included: list[Literal["ids", "embeddings", "metadatas", "documents"]]
    

    def rerank(self, new_order: Sequence[int]) -> Self:
        old = {attr: value.copy() for attr in self.included if (value := getattr(self, attr)) is not None}
        for attr in self.included:
            new = getattr(self, attr)
            for new_index, old_index in enumerate(new_order):
                new[new_index] = old[attr][old_index]
        return self
    

    def trim(self, start: Optional[int] = None, end: Optional[int] = None) -> Self:
        for attr in self.included:
            setattr(self, attr, getattr(self, attr)[start:end])
        return self


    def reduce_to_top_n(self, n: int) -> Self:
        return self.trim(end=n)
    

    def sort(
            self,
            by: Literal["ids", "embeddings", "documents", "metadatas", "distances"] = "ids",
            key: Optional[Callable] = None,
            reverse: bool = False
    ) -> Self:
        
        _key = (lambda x: key(x[1])) if key else (lambda x: x[1])
        new_order = [x[0] for x in sorted(enumerate(getattr(self, by)), key=_key, reverse=reverse)]
        return self.rerank(new_order)
        
    
    def zip(self, *include: Literal["ids", "embeddings", "metadatas", "documents"]):
        return zip(*(getattr(self, attr) for attr in include or self.included))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Document:
        return Document(
            id=self.ids[index],
            embedding=self.embeddings[index] if self.embeddings else None,
            metadata=self.metadatas[index] if self.metadatas else None,
            text=self.documents[index] if self.documents else None
        )

    def __iter__(self) -> Iterator[Document]:
        return iter(self[i] for i in range(len(self)))
    
    def list(self) -> list[Document]:
        return list(self)
    

class QueryResult(GetResult):
    distances: Optional[list[float]]
    included: list[Literal["ids", "embeddings", "metadatas", "documents", "distances"]]


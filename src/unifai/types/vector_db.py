from typing import Optional, Sequence, Literal
from pydantic import BaseModel, RootModel

from .embeddings import Embeddings, Embedding

class VectorDBGetResult(BaseModel):
    ids: list[str]
    embeddings: Optional[list[Embedding]]
    documents: Optional[list[str]]
    metadatas: Optional[list[dict]]    
    included: Sequence[Literal["embeddings", "metadatas", "documents"]]

# class VectorDBQueryResult(VectorDBGetResult):
#     distances: Optional[list[float]]
#     included: Sequence[Literal["embeddings", "metadatas", "documents", "distances"]]

class VectorDBQueryResult(VectorDBGetResult):
    distances: Optional[list[float]]
    included: Sequence[Literal["embeddings", "metadatas", "documents", "distances"]]

class VectorDBQueryResults(RootModel[list[VectorDBQueryResult]]):
    pass
    # results: list[VectorDBQueryResult]
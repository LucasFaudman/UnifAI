from typing import Optional, Literal, Union, Self, Any

from pydantic import BaseModel

from .embeddings import Embedding
from .response_info import ListWithResponseInfo

class Document(BaseModel):
    id: str
    text: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    embedding: Optional[Embedding] = None

class Documents(ListWithResponseInfo[Document]):
    pass

class QueryDocument(Document):
    query: Optional[str] = None
    rank: Optional[int] = None
    distance: Optional[float] = None

class QueryDocuments(ListWithResponseInfo[QueryDocument]):
    pass

class RerankedDocument(QueryDocument):
    original_rank: int
    similarity: float

class RerankedDocuments(ListWithResponseInfo[RerankedDocument]):
    pass


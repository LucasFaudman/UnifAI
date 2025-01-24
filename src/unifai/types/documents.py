from typing import Optional, Literal, Union, Self, Any

from ._base_model import BaseModel
from .embeddings import Embedding
from .response_info import ListWithResponseInfo

class Document(BaseModel):
    """
    A document with an id and optional text, metadata, and embeddings.

    Args:
        id (str): The document ID.
        metadata (Optional[dict[str, Any]]): The document metadata. Defaults to None.
        text (Optional[str]): The document text. Defaults to None.
        embedding (Optional[Embedding]): The document embedding. Defaults to None.
    """
    id: str
    metadata: Optional[dict[str, Any]] = None
    text: Optional[str] = None
    embedding: Optional[Embedding] = None

    def __len__(self) -> int:
        return self.length

    @property
    def length(self) -> int:
        return len(self.text) if self.text else 0
    
    @property
    def embedding_dimensions(self) -> int:
        return len(self.embedding) if self.embedding else 0

class RankedDocument(Document):
    """
    A ranked document with an id, rank, distance, and optional text, metadata, embeddings and query.
    
    Args:
        id (str): The document chunk ID.
        metadata (Optional[dict[str, Any]]): The document chunk metadata. Defaults to None.
        text (Optional[str]): The document chunk text. Defaults to None.
        embedding (Optional[Embedding]): The document chunk embedding. Defaults to None.
        rank (int): The document rank.
        distance (Optional[float]): The document distance. Defaults to None.
        query (Optional[str|Embedding]): The query. Defaults to None.        
    """
    rank: int
    distance: Optional[float] = None
    query: Optional[str]|Embedding = None


class RerankedDocument(RankedDocument):
    """
    A reranked document with an id, rank, distance, original rank, similarity_score, and optional text, metadata, embeddings and query.

    Args:
        id (str): The document chunk ID.
        metadata (Optional[dict[str, Any]]): The document chunk metadata. Defaults to None.
        text (Optional[str]): The document chunk text. Defaults to None.
        embedding (Optional[Embedding]): The document chunk embedding. Defaults to None.
        rank (int): The document rank.
        distance (Optional[float]): The document distance. Defaults to None.
        query (Optional[str|Embedding]): The query. Defaults to None.
        # original_rank (int): The original document rank.
        similarity_score (float): The document similarity_score.
    """
    # original_rank: int
    similarity_score: float

class DocumentChunk(Document):
    """
    A document chunk with an id, parent document id, and optional text, metadata, 
    embeddings, chunk size, start and end indices.

    Args:
        id (str): The document chunk ID.
        metadata (Optional[dict[str, Any]]): The document chunk metadata. Defaults to None.
        text (Optional[str]): The document chunk text. Defaults to None.
        embedding (Optional[Embedding]): The document chunk embedding. Defaults to None.
        source_id (str): The parent document ID.        
        chunk_size (Optional[int]): The document chunk size. Defaults to None.
        start_index (Optional[int]): The start index of the document chunk. Defaults to None.
        end_index (Optional[int]): The end index of the document chunk. Defaults to None.
    """
    source_id: str
    chunk_size: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None

class RankedDocumentChunk(DocumentChunk, RankedDocument):
    """
    A ranked document chunk with an id, rank, distance, parent document id, and optional text, metadata, 
    embeddings, chunk size, start and end indices.

    Args:
        id (str): The document chunk ID.
        metadata (Optional[dict[str, Any]]): The document chunk metadata. Defaults to None.
        text (Optional[str]): The document chunk text. Defaults to None.
        embedding (Optional[Embedding]): The document chunk embedding. Defaults to None.
        rank (int): The document rank.
        distance (Optional[float]): The document distance. Defaults to None.
        query (Optional[str|Embedding]): The query. Defaults to None.        
        source_id (str): The parent document ID.        
        chunk_size (Optional[int]): The document chunk size. Defaults to None.
        start_index (Optional[int]): The start index of the document chunk. Defaults to None.
        end_index (Optional[int]): The end index of the document chunk. Defaults to None.
    """

class RerankedDocumentChunk(RankedDocumentChunk, RerankedDocument):
    """
    A reranked document chunk with an id, rank, distance, original rank, similarity_score, parent document id, and optional text, metadata, 
    embeddings, chunk size, start and end indices.

    Args:
        id (str): The document chunk ID.
        metadata (Optional[dict[str, Any]]): The document chunk metadata. Defaults to None.
        text (Optional[str]): The document chunk text. Defaults to None.
        embedding (Optional[Embedding]): The document chunk embedding. Defaults to None.
        rank (int): The document rank.
        distance (Optional[float]): The document distance. Defaults to None.
        query (Optional[str|Embedding]): The query. Defaults to None.
        # original_rank (int): The original document rank.
        similarity_score (float): The document similarity_score.
        source_id (str): The parent document ID.        
        chunk_size (Optional[int]): The document chunk size. Defaults to None.
        start_index (Optional[int]): The start index of the document chunk. Defaults to None.
        end_index (Optional[int]): The end index of the document chunk. Defaults to None.
    """

class Documents(ListWithResponseInfo[Document]):
    """
    A list of Document objects with ResponseInfo.
    """   
class RankedDocuments(ListWithResponseInfo[RankedDocument]):
    """
    A list of QueryDocument objects with ResponseInfo.
    """
class RerankedDocuments(ListWithResponseInfo[RerankedDocument]):
    """
    A list of RerankedDocument objects with ResponseInfo.
    """
class DocumentChunks(ListWithResponseInfo[DocumentChunk]):
    """
    A list of DocumentChunk objects with ResponseInfo.
    """
class RankedDocumentChunks(ListWithResponseInfo[RankedDocumentChunk]):
    """
    A list of RankedDocumentChunk objects with ResponseInfo.
    """
class RerankedDocumentChunks(ListWithResponseInfo[RerankedDocumentChunk]):
    """
    A list of RerankedDocumentChunk objects with ResponseInfo.
    """



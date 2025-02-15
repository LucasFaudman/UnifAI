from typing import Iterable, Optional, Collection, Sequence
from itertools import zip_longest
from ...types.documents import (
    Document, Documents, 
    RankedDocument, RankedDocuments, 
    RerankedDocument, RerankedDocuments,
    DocumentChunk, DocumentChunks,
    RankedDocumentChunk, RankedDocumentChunks,
    RerankedDocumentChunk, RerankedDocumentChunks
)

from ...utils.iter_utils import zippable

def documents_to_lists(
        documents: Iterable[Document] | Documents,
        attrs: Sequence[str] = ("ids", "texts", "metadatas", "embeddings")
) -> tuple[list, ...]:
    if not documents:
        raise ValueError("No documents provided")    
    _lists = {attr: [] for attr in attrs}
    for document in documents:
        for attr in attrs:
            value = getattr(document, attr[:-1])
            if value is not None:
                _lists[attr].append(value)
            else:
                raise ValueError(f"All documents must have {attr}. Got {value=} for Document {document.id=}")
    return tuple(_lists.values())

def iterables_to_documents(
        *iterables: Optional[Iterable],
        attrs: Sequence[str] = ("id", "metadata", "text")
) -> Iterable[Document]:
    for _values in zip_longest(*zippable(*iterables)): 
        yield Document(**dict(zip(attrs, _values)))


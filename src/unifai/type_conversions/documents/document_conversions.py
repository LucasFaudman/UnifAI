from typing import Iterable, Optional, Collection
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

# def documents_to_lists(
#         documents: Iterable[Document] | Documents
# ) -> tuple[list[str], Optional[list[dict]], Optional[list[str]], list[Embedding]]:
#     if not documents:
#         raise ValueError("No documents provided")
#     if isinstance(documents, Documents):
#         documents = documents.list()
#     if not isinstance(documents, list) or not isinstance((doc0 := documents[0]), Document):
#         raise TypeError(f"Invalid documents type: {type(documents)}. Must be list of Document or a Documents object")
    
#     ids = []
#     metadatas = texts = embeddings = None
#     if doc0.metadata is not None:
#         metadatas = []
#     if doc0.embedding is not None:
#         embeddings = []
#     elif doc0.text is not None:
#         texts = []

#     for document in documents:
#         if not document.id:
#             raise ValueError(f"All documents must have an id. Got {document.id=}")
#         ids.append(document.id)

#         if metadatas is not None:
#             if document.metadata is None:
#                 raise ValueError(f"All documents must have metadata when the first document has metadata. Got {document.metadata=}")
#             metadatas.append(document.metadata)

#         if texts is not None:
#             if document.text is None:
#                 raise ValueError(f"All documents must have text when the first document has text. Got {document.text=}")
#             texts.append(document.text)              

#         if embeddings is not None:
#             if document.embedding is None:
#                 raise ValueError(f"All documents must have an embedding when the first document has an embedding. Got {document.embedding=}")
#             embeddings.append(document.embedding)           

#     # if len(ids) != (len_documents := len(documents)):
#     #     raise ValueError("All documents must have an id")
#     # if metadatas is not None and len(metadatas) != len_documents:
#     #     raise ValueError("All documents must have metadata")
#     # if texts is not None and len(texts) != len_documents:
#     #     raise ValueError("All documents must have text")        
#     # if embeddings is not None and len(embeddings) != len_documents:
#     #     raise ValueError("All documents must have an embedding")        

#     return ids, metadatas, texts, embeddings

def documents_to_lists(
        documents: Iterable[Document] | Documents,
        attrs: tuple|list[str] = ("ids", "texts", "metadatas", "embeddings")
        # *attrs: str
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
        attrs: tuple|list[str] = ("id", "metadata", "text")
) -> Iterable[Document]:
    for _values in zip_longest(*zippable(*iterables)): 
        yield Document(**dict(zip(attrs, _values)))


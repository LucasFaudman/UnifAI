from typing import Optional, Sequence, Literal, Self, Callable, Iterator, Iterable, Any, ClassVar, Type, TypeVar, Generic
from ._base_model import BaseModel

from ..type_conversions.documents import documents_to_lists
from .documents import Document, Documents, DocumentChunk, DocumentChunks, RankedDocument, RankedDocuments, RerankedDocument, RerankedDocuments
from .embeddings import Embeddings, Embedding


from itertools import zip_longest

class GetResult(BaseModel):
    ids: list[str]
    metadatas: Optional[list[dict|None]] = None
    texts: Optional[list[str]] = None
    embeddings: Optional[list[Embedding|None]] = None
    included: list[Literal["ids", "metadatas", "texts", "embeddings"]]

    _document_class: ClassVar = Document
    _default_sorted_by: ClassVar[Optional[tuple[str, bool]]] = None
        
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
    
    @property
    def sorted_by(self) -> Optional[tuple[str, bool]]:
        if hasattr(self, "_sorted_by"):
            return self._sorted_by
        self._sorted_by = self._default_sorted_by
        return self._sorted_by

    def sort(
            self,
            by: Literal["ids", "metadatas", "texts", "embeddings", "distances", "similarity_scores"] = "ids",
            key: Optional[Callable] = None,
            reverse: bool = False
    ) -> Self:       
        _key = (lambda x: key(x[1])) if key else (lambda x: x[1])
        new_order = [x[0] for x in sorted(enumerate(getattr(self, by)), key=_key, reverse=reverse)]
        self._sorted_by = by, reverse
        return self.rerank(new_order)
    
    def yield_attrs(self, *include: Literal["ids", "metadatas", "texts", "embeddings", "distances"]) -> Iterator:
        _include = include or self.included
        for attr in _include:
            yield getattr(self, attr)

    def get_attrs(self, *include: Literal["ids", "metadatas", "texts", "embeddings", "distances"]) -> tuple:
        return tuple(self.yield_attrs(*include))

    def iterables(self, *include: Literal["ids", "metadatas", "texts", "embeddings", "distances"]) -> Iterator[Iterable]:
        for _value in self.yield_attrs(*include):
            yield _value or ()
            
    def zip(self, *include: Literal["ids", "metadatas", "texts", "embeddings", "distances"], fillvalue: Optional[Any]=None) -> Iterator[tuple]:
        return zip_longest(*self.iterables(*include), fillvalue=fillvalue)
    
    def zipped_dicts(self, *include: Literal["ids", "metadatas", "texts", "embeddings", "distances"], fillvalue: Optional[Any]=None) -> Iterator[dict]:
        _include = include or self.included
        _doc_keys = [attr[:-1] for attr in _include]
        for values in self.zip(*_include, fillvalue=fillvalue):
            yield dict(zip(_doc_keys, values))
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Document:
        return self._document_class(**{attr[:-1]: getattr(self, attr)[index] for attr in self.included})

    def __iter__(self) -> Iterator[Document]:
        for doc_dict in self.zipped_dicts(*self.included):
            yield self._document_class(**doc_dict)

    def to_documents(self) -> list[Document]:
        return list(self)

    @classmethod
    def from_documents(cls, 
                       documents: Iterable[Document]|Documents, 
                       included: list[Literal["ids", "metadatas", "texts", "embeddings"]] = ["ids", "metadatas", "texts", "embeddings"]                       
                       ) -> Self:
                        
        _lists = documents_to_lists(documents, included)
        _doc_keys = [attr[:-1] for attr in included]
        return cls(**dict(zip(_doc_keys, _lists)), included=included)
    
    
class QueryResult(GetResult):
    distances: Optional[list[float]]
    query: Optional[str|Embedding]
    included: list[Literal["ids", "embeddings", "metadatas", "texts", "distances"]]

    _document_class: ClassVar = RankedDocument
    _default_sorted_by: ClassVar[Optional[tuple[str, bool]]] = "distances", False

    @classmethod
    def from_get_result(cls, 
                        get_result: GetResult, 
                        distances: list[float],
                        query: Optional[str|Embedding] = None, 
                        ) -> Self:
        return cls(
            ids=get_result.ids,
            metadatas=get_result.metadatas,
            texts=get_result.texts,
            embeddings=get_result.embeddings,
            distances=distances,
            query=query,
            included=[*get_result.included, "distances"]
        ).sort(by="distances", reverse=False)

    def __getitem__(self, index: int) -> RankedDocument:
        return RankedDocument(query=self.query, **{attr[:-1]: getattr(self, attr)[index] for attr in self.included})

    def __iter__(self) -> Iterator[RankedDocument]:
        for rank, doc_dict in enumerate(self.zipped_dicts()):
            yield RankedDocument(query=self.query, rank=rank, **doc_dict)

    def to_documents(self) -> list[RankedDocument]:
        return list(self)
    
    @classmethod
    def from_documents(cls, 
                       documents: Iterable[RankedDocument]|RankedDocuments, 
                       included: list[Literal["ids", "metadatas", "texts", "embeddings", "distances"]] = ["ids", "metadatas", "texts", "embeddings", "distances"]                       
                       ) -> Self:
        return super().from_documents(documents, included)
    
    def trim_by_distance(self, max_distance: float) -> Self:
        """Remove all documents with distance > max_distance"""

        # raise error if distances not included in result
        if not (distances := self.distances):
            raise ValueError("Distances not included in result")
        
        # sort by distances in ascending order if not already sorted
        if not self.sorted_by == ("distances", False):
            self.sort(by="distances", reverse=False)
                
        # binary search for first index with distance > max_distance
        start = 0
        end = len(distances)
        while start < end:
            mid = (start + end) // 2
            if distances[mid] <= max_distance:
                start = mid + 1
            else:
                end = mid
        return self.trim(end=start)

class RerankedQueryResult(QueryResult):
    similarity_scores: Optional[list[float]]
    included: list[Literal["ids", "embeddings", "metadatas", "texts", "distances", "similarity_scores"]]

    _document_class: ClassVar = RerankedDocument
    _default_sorted_by: ClassVar[Optional[tuple[str, bool]]] = "similarity_score", True

    @classmethod
    def from_query_result(cls, query_result: QueryResult, similarity_scores: list[float]) -> Self:
        return cls(
            ids=query_result.ids,
            metadatas=query_result.metadatas,
            texts=query_result.texts,
            embeddings=query_result.embeddings,
            distances=query_result.distances,
            query=query_result.query,
            similarity_scores=similarity_scores,
            included=[*query_result.included, "similarity_scores"]
        ).sort(by="similarity_scores", reverse=True)

    def __getitem__(self, index: int) -> RerankedDocument:
        return RerankedDocument(query=self.query, rank=index, **{attr[:-1]: getattr(self, attr)[index] for attr in self.included})
    
    def __iter__(self) -> Iterator[RerankedDocument]:
        for rank, doc_dict in enumerate(self.zipped_dicts()):
            yield RerankedDocument(query=self.query, rank=rank, **doc_dict)

    def to_documents(self) -> list[RerankedDocument]:
        return list(self)
    
    @classmethod
    def from_documents(cls, 
                          documents: Iterable[RerankedDocument]|RerankedDocuments, 
                          included: list[Literal["ids", "metadatas", "texts", "embeddings", "distances", "similarity_scores"]] = ["ids", "metadatas", "texts", "embeddings", "distances", "similarity_scores"]                       
                          ) -> Self:
          return super().from_documents(documents, included)
    
    def trim_by_similarity_score(self, min_similarity_score: float) -> Self:
        """Remove all documents with similarity_score < min_similarity_score"""

        # raise error if similarity_scores not included in result
        if not (similarity_scores := self.similarity_scores):
            raise ValueError("Similarity scores not included in result")
        
        # sort by similarity_scores in descending order if not already sorted
        if not self.sorted_by == ("similarity_scores", True):
            self.sort(by="similarity_scores", reverse=True)
                
        # binary search for first index with similarity_score < min_similarity_score
        start = 0
        end = len(similarity_scores)
        while start < end:
            mid = (start + end) // 2
            if similarity_scores[mid] >= min_similarity_score:
                start = mid + 1
            else:
                end = mid
        return self.trim(end=start)
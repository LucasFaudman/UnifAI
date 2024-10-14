from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from .._core._base_vector_db_index import VectorDBIndex
from .._core._base_reranker import Reranker
from ..client.specs import RAGSpec, FuncSpec
from ..components.prompt_template import PromptTemplate


class Retriever:
    def query(self, query_text: str, n_results: int, **kwargs) -> VectorDBQueryResult:
        raise NotImplementedError
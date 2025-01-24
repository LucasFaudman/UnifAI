from .._base_components._base_ragpipe import BaseRAGPipe
from ...configs.rag_config import RAGConfig, LoaderInputP, QueryInputP
from ...types.annotations import NewInputP
from ...types.db_results import QueryResult
from ...types.documents import Document, Documents
from ...types.response_info import ResponseInfo
from typing import Generic, Concatenate, cast, Callable, Iterable, Generator

class RAGPipe(BaseRAGPipe[RAGConfig[LoaderInputP, QueryInputP], LoaderInputP, QueryInputP], Generic[LoaderInputP, QueryInputP]):
    component_type = "ragpipe"
    provider = "default"
    config_class = RAGConfig

    def set_document_loader(self, document_loader: Callable[NewInputP, Documents | Iterable[Document] | Generator[Document, None, ResponseInfo | None]]):
        self._set_document_loader(document_loader)
        return cast("RAGPipe[NewInputP, QueryInputP]", self)

    def set_query_modifier(self, query_modifier: Callable[NewInputP, str|Callable[..., str]]):
        self._set_query_modifier(query_modifier)
        return cast("RAGPipe[LoaderInputP, NewInputP]", self)
    
    def set_prompt_template(self, prompt_template: Callable[Concatenate[QueryResult, NewInputP], str|Callable[..., str]]):
        self._set_prompt_template(prompt_template)
        return cast("RAGPipe[LoaderInputP, NewInputP]", self)    
    
    def set_query_modifier_and_prompt_template(
            self,
            query_modifier: Callable[NewInputP, str|Callable[..., str]],
            prompt_template: Callable[Concatenate[QueryResult, NewInputP], str|Callable[..., str]],
            ):
        self._set_query_modifier(query_modifier)
        return self.set_prompt_template(prompt_template)
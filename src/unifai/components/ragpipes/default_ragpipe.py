from .._base_components._base_ragpipe import BaseRAGPipe
from ...configs.rag_config import RAGConfig
from ...types.annotations import InputP, NewInputP
from ...types.db_results import QueryResult
from typing import Generic, Concatenate, cast, Callable

class RAGPipe(BaseRAGPipe[RAGConfig, InputP], Generic[InputP]):
    component_type = "ragpipe"
    provider = "default"
    config_class = RAGConfig

    def set_query_modifier(self, query_modifier: Callable[NewInputP, str|Callable[..., str]]):
        self._set_query_modifier(query_modifier)
        return cast("RAGPipe[NewInputP]", self)
    
    def set_prompt_template(self, prompt_template: Callable[Concatenate[QueryResult, NewInputP], str|Callable[..., str]]):
        self._set_prompt_template(prompt_template)
        return cast("RAGPipe[NewInputP]", self)    
    
    def set_query_modifier_and_prompt_template(
            self,
            query_modifier: Callable[NewInputP, str|Callable[..., str]],
            prompt_template: Callable[Concatenate[QueryResult, NewInputP], str|Callable[..., str]],
            ):
        self._set_query_modifier(query_modifier)
        return self.set_prompt_template(prompt_template)    
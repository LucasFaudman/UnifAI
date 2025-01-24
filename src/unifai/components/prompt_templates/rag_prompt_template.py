from typing import Optional, Callable, Any
from ...types import QueryResult
from .prompt_template import PromptTemplate
from pydantic import Field

class RAGPromptTemplate(PromptTemplate):
    "{query}\n\nCONTEXT:\n\n{result}"
    value_formatters: dict[str|type, Optional[Callable[..., Any]]] = Field(default={ 
            QueryResult: lambda result: "\n".join(f"DOCUMENT: {doc.id}\n{doc.text}\n" for doc in result)
    }, exclude=True)
    
    __init__ = PromptTemplate.__init__

    def __call__(
        self, 
        result: QueryResult, 
        *args,
        **kwargs
        ) -> str:
        return super().__call__(*args, **kwargs, result=result)
    
    
default_rag_prompt_template = RAGPromptTemplate("{query}\n\nCONTEXT:\n\n{result}")
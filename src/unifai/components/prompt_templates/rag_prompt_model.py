from typing import Optional
from ...types import QueryResult
from .prompt_model import PromptModel

class RAGPromptModel(PromptModel):
    "{query}\n\nCONTEXT:\n{result}"    
    result: QueryResult
    query: Optional[str] = ""
    value_formatters = { 
        QueryResult: lambda result: "\n".join(f"DOCUMENT: {doc.id}\n{doc.text}\n" for doc in result)
    }

    def __init__(self, result: QueryResult, *args, **kwargs):
        kwargs["result"] = result
        super().__init__(*args, **kwargs)

    def __call__(
        self, 
        result: QueryResult, 
        *args,
        **kwargs
        ) -> str:
        return super().__call__(*args, **kwargs, result=result)
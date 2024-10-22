from typing import Optional, Union, Sequence, Any, Literal, Callable, Mapping, Type
from pydantic import BaseModel, Field, ConfigDict

from ..types.valid_inputs import LLMProvider, VectorDBProvider, EmbeddingProvider, ToolInput, RerankProvider
from ..types.message import Message
from ..types.tool import Tool
from ..types.vector_db import VectorDBQueryResult
from ..components import PromptTemplate, ToolCaller, DocumentDB


class Spec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "default_spec"

    def with_spec(self, **kwargs):
        return self.model_copy(update=kwargs)


def default_query_result_formatter(result: VectorDBQueryResult) -> str:
    return "\n".join(f"DOCUMENT: {id}\n{doc}" for id,doc in result.zip("ids", "documents"))

DEFAULT_RAG_PROMPT_TEMPLATE = PromptTemplate(
    "{prompt_header}{query}{sep}{result_header}{query_result}{response_start}",
    value_formatters={VectorDBQueryResult: default_query_result_formatter},
    prompt_header="PROMPT:\n",
    result_header="CONTEXT:\n",
    sep="\n\n",
    response_start="\n\nRESPONSE: ",
)       

class RAGSpec(Spec):
    name: str = "default_rag_spec"
    index_name: str = "default_index"

    vector_db_provider: Optional[VectorDBProvider] = None
    retriever_kwargs: dict[str, Any] = Field(default_factory=dict)
    
    embedding_provider: Optional[EmbeddingProvider] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    embedding_distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None
    
    rerank_provider: Optional[RerankProvider] = None
    rerank_model: Optional[str] = None
    reranker_kwargs: dict[str, Any] = Field(default_factory=dict)

    top_n: int = 10
    top_k: Optional[int] = 30
    where: Optional[dict] = None
    where_document: Optional[dict] = None
    document_db_class_or_instance: Optional[Type[DocumentDB]|DocumentDB] = None
    document_db_kwargs: dict[str, Any] = Field(default_factory=dict)

    prompt_template: PromptTemplate = PromptTemplate(
        "{prompt_header}{query}{sep}{result_header}{query_result}{response_start}",
        value_formatters={
            VectorDBQueryResult: 
            lambda result: "\n".join(f"DOCUMENT: {id}\n{doc}" for id, doc in result.zip("ids", "documents"))
        },
        prompt_header="PROMPT:\n",
        result_header="CONTEXT:\n",        
        response_start="\n\nRESPONSE: ",
        sep="\n\n",        
    )
    prompt_template_kwargs: dict[str, Any] = Field(default_factory=dict)

        


class FuncSpec(Spec):
    name: str = "default_eval_spec"
    provider: Optional[LLMProvider] = None           
    model: Optional[str] = None

    prompt_template: PromptTemplate|str = PromptTemplate("{content}", value_formatters={Message: lambda m: m.content})
    prompt_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    rag_spec: Optional[RAGSpec|str] = None
    
    system_prompt: Optional[str|PromptTemplate|Callable[...,str]] = None
    system_prompt_kwargs: dict[str, Any] = Field(default_factory=dict)

    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None   
    response_format: Optional[Literal["text", "json", "json_schema"]] = None
    return_on: Union[Literal["content", "tool_call", "message"], str, Tool, list[str|Tool]] = "content"
    return_as: Literal["self", 
                       "messages", 
                       "last_message", 
                       "last_content",
                       "last_tool_call",
                       "last_tool_call_args",
                       "last_tool_calls", 
                       "last_tool_calls_args"
                       ] = "self"
    reset_on_return: bool = False
    output_parser: Optional[Callable|Type[BaseModel]|BaseModel] = None
    output_parser_kwargs: dict[str, Any] = Field(default_factory=dict)

    tools: Optional[Sequence[ToolInput]] = None            
    tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None
    enforce_tool_choice: bool = True
    tool_choice_error_retries: int = 3
    tool_callables: Optional[dict[str, Callable]] = None
    tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller
    tool_caller_kwargs: dict[str, Any] = Field(default_factory=dict)

    max_messages_per_run: int = 10
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    error_handlers: Optional[Mapping[Type[Exception], Callable[..., Any]]] = None


class AgentSpec(Spec):
    name: str = "default_agent_spec"
    ai_functions: list[FuncSpec|str] = Field(default_factory=list)

eval_spec = FuncSpec()
rag_spec = RAGSpec()
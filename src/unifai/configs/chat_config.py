from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, BaseModel
from ..types import (
    Message,
    Tool,
)
from ..components.prompt_template import PromptTemplate
from ._base_configs import ComponentConfig
from .llm_config import LLMConfig
from .tool_caller_config import ToolCallerConfig

class ChatConfig(ComponentConfig):
    # name: ClassVar[str] = "chat"
    component_type: ClassVar = "chat"
    provider: ClassVar[str] = "default" 

    llm: ProviderName | LLMConfig | tuple[ProviderName, ComponentName] = "default"
    llm_model: Optional[ModelName] = None

    system_prompt: Optional[str | PromptTemplate | Callable[...,str]] = None
    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None
    # prompt_template: PromptTemplate | str | Callable[..., str] = PromptTemplate("{content}", value_formatters={Message: lambda m: m.content})
    # rag_config: Optional[RAGConfig | str] = None

    
    tools: Optional[list[ToolInput]] = None            
    tool_choice: Optional[ToolName | Tool | Literal["auto", "required", "none"] | list[ToolName | Tool | Literal["auto", "required", "none"]]] = None
    enforce_tool_choice: bool = True
    tool_choice_error_retries: int = 3
    tool_callables: Optional[dict[ToolName, Callable[..., Any]]] = None
    tool_caller: Optional[ProviderName | ToolCallerConfig | tuple[ProviderName, ComponentName]] = None

    # response_format: Optional[Literal["text", "json"] | Type[BaseModel] | Tool | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]] = None
    response_format: Optional[Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]] = None
    return_on: Union[Literal["content", "tool_call", "message"], ToolName, Tool, list[ToolName | Tool]] = "content"

    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    max_messages_per_run: int = 10
    max_tool_calls_per_run: Optional[int] = None
    max_tokens_per_run: Optional[int] = None
    max_input_tokens_per_run: Optional[int] = None
    max_output_tokens_per_run: Optional[int] = None
    count_tokens_proactively: bool = False    


    error_retries: dict[Literal[
        "api_error",
        "content_filter_error",
        "tool_choice_error", 
        "tool_call_argument_validation_error",
        "tool_call_execution_error",
        "tool_call_timeout_error",
    ], int] = {
        "api_error": 0,
        "content_filter_error": 0,
        "tool_choice_error": 3, 
        "tool_call_argument_validation_error": 3,
        "tool_call_execution_error": 0,
        "tool_call_timeout_error": 0,
    }
    error_handlers: dict[Literal[
        "api_error",
        "content_filter_error",
        "tool_choice_error", 
        "tool_call_argument_validation_error",
        "tool_call_execution_error",
        "tool_call_timeout_error",
    ], Callable[..., Any]] = {}

    extra_kwargs: Optional[dict[Literal["chat", "system_prompt", "run", "run_stream"], dict[str, Any]]] = None


DEFAULT_CHAT_CONFIG = ChatConfig()
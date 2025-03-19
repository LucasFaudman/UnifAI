from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, Generic

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, BaseModel, InputP, MessageInput
from ..types import (
    Message,
    Tool,
)
from ..exceptions import UnifAIError, ToolChoiceError
from ..components._base_components._base_prompt_template import PromptModel
from ..components._base_components._base_tool_caller import ToolCaller
from ..components._base_components._base_llm import LLM

from ._base_configs import ComponentConfig, _ComponentEndMixin
from .llm_config import LLMConfig
from .tool_caller_config import ToolCallerConfig

ChatExtraKwargsKeys = Literal["chat", "system_prompt", "run", "run_stream"]

class _ChatConfig(ComponentConfig, Generic[InputP]):
    component_type: ClassVar = "chat"
    provider: ClassVar[str] = "default" 

    llm: LLM | LLMConfig | ProviderName | tuple[ProviderName, ComponentName] = "default"
    llm_model: Optional[ModelName] = None

    system_prompt: Optional[str | Callable[..., str] | PromptModel | Type[PromptModel]] = None
    system_prompt_kwargs: Optional[dict[str, Any]] = None
    examples: Optional[list[Message | dict[Literal["input", "response"], Any]]] = None
    # initial_messages: Optional[Sequence[MessageInput]] = None
    
    tools: Optional[list[ToolInput]] = None
    tool_choice: Optional[ToolName | Tool | Literal["auto", "required", "none"] | list[ToolName | Tool | Literal["auto", "required", "none"]]] = None
    enforce_tool_choice: bool = True
    tool_callables: Optional[dict[ToolName, Callable[..., Any]]] = None
    tool_caller: Optional[ToolCaller | ToolCallerConfig | ProviderName | tuple[ProviderName, ComponentName]] = "default"

    response_format: Optional[Literal["text", "json"] | dict[str, Any] | Type[BaseModel] | Tool] = None
    return_on: Literal["content", "tool_call", "message"] | ToolName | Tool | list[ToolName | Tool] = "content"

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

    # error_retries: dict[Type[UnifAIError | Exception], int] = {
    #     ToolChoiceError: 3,
    # }
    # error_handlers: dict[Type[UnifAIError | Exception], Callable] = {}

    # extra_kwargs: Optional[dict[Literal["chat", "system_prompt", "run", "run_stream"], dict[str, Any]]] = None

class ChatConfig(
    _ComponentEndMixin[Literal["chat", "system_prompt", "run", "run_stream"]], 
    _ChatConfig[InputP], 
    Generic[InputP]
    ):
    """"""

ChatConfig()
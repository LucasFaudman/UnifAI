from typing import TYPE_CHECKING, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, ResponseFormatInput
from ..types import (
    Message,
    Tool,
    ToolCall,
)


from ..components.prompt_template import PromptTemplate
from ..components.output_parsers.pydantic_output_parser import PydanticParser
from ._base_configs import BaseConfig
from .chat_config import ChatConfig
from .llm_config import LLMConfig
from .rag_config import RAGConfig
from .tool_caller_config import ToolCallerConfig

from typing import TypeAlias, TypeVar, Generic
if TYPE_CHECKING:
    from ..components.chat import Chat

from pydantic import BaseModel, Field
InputT = TypeVar('InputT', bound=BaseModel) # Input type of the function
OutputT = TypeVar('OutputT', "Chat", Message, ToolCall, str) # Type of the output of the function to be parsed by the output_parser
ReturnT = TypeVar('ReturnT', bound=BaseModel) # Return type of the function (the final output type of the parsed output)

class _FunctionConfig(BaseConfig, Generic[InputT, OutputT, ReturnT]):
    # model_config = ConfigDict(arbitrary_types_allowed=True)    
    stateless: bool = True

    llm: Optional[LLMConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    llm_model: Optional[ModelName] = None

    system_prompt: Optional[str | PromptTemplate | Callable[...,str]] = None
    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None
    prompt_template: str | PromptTemplate | Callable[..., str] = "{input}" #PromptTemplate("{content}", value_formatters={Message: lambda m: m.content})
    rag_config: Optional[RAGConfig | ComponentName] = None
    # response_format: Optional[Literal["text", "json"] | Type[BaseModel] | Tool | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]] = None
    response_format: Optional[Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]] = None

    return_on: Union[Literal["content", "tool_call", "message"], ToolName, Tool, list[ToolName | Tool]] = "content"      
    # return_as: Literal["self", 
    #                    "messages", 
    #                    "last_message",
    #                    "last_content",
    #                    "last_tool_call",
    #                    "last_tool_call_args",
    #                    "last_tool_calls", 
    #                    "last_tool_calls_args"
    #                    ] = "self"

    input_parser: Callable[[InputT], dict|str] = lambda x: x.model_dump() if isinstance(x, BaseModel) else x
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT]
    exception_handlers: Optional[dict[Type[Exception], Callable[..., ReturnT]]] = None

    # @property
    # def input_type(self) -> Type[InputT]:
    #     return self.input_parser.__annotations__.get("input", BaseModel)
    
    @property
    def output_type(self) -> Type[OutputT]:
        return self.output_parser.__annotations__.get("output", Message)
    
    # @property
    # def return_type(self) -> Type[ReturnT]:
    #     return self.output_parser.__annotations__.get("return", Message)

class FunctionConfig(ChatConfig, _FunctionConfig[InputT, OutputT, ReturnT], Generic[InputT, OutputT, ReturnT]):
    input_parser: Callable[[InputT], dict|str] = Field(default=Message.get_content)
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] = Field(default=PydanticParser(Message, Message))


FunctionConfig()
from typing import TYPE_CHECKING, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, ResponseFormatInput, InputP
from ..types import (
    Message,
    Tool,
    ToolCall,
)


from ..components.prompt_templates import PromptTemplate
from ._base_configs import BaseConfig
from .chat_config import ChatConfig
from .llm_config import LLMConfig
from .rag_config import RAGConfig
from .tool_caller_config import ToolCallerConfig
from .input_parser_config import InputParserConfig, InputP, InputReturnT
from .output_parser_config import OutputParserConfig, OutputT, ReturnT
from .rag_config import RAGConfig

from typing import TypeAlias, TypeVar, Generic
if TYPE_CHECKING:
    from ..components.chats import Chat

from pydantic import BaseModel, Field
# InputT = TypeVar('InputT', bound=BaseModel) # Input type of the function

def convert_input_to_user_message(input: Message|str) -> Message:
    if isinstance(input, Message):
        return input if input.role == "user" else Message(content=input.content, role="user")
    return Message(content=input)
    
# def return_last_message(output: "Chat") -> Message:
#     return output.last_message or Message(content="")

def return_last_message(output: Message) -> Message:
    return output.last_message or Message(content="")

class _FunctionConfig(BaseConfig, Generic[InputP, InputReturnT, OutputT, ReturnT]):
    component_type: ClassVar = "function"
    provider: ClassVar[str] = "default" 

    stateless: bool = True
    
    # llm: Optional[LLMConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    # llm_model: Optional[ModelName] = None

    # system_prompt: Optional[str | PromptTemplate | Callable[...,str]] = None
    # examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None

    input_parser: Callable[InputP, InputReturnT | Callable[..., InputReturnT]] | InputParserConfig[InputP, InputReturnT] | RAGConfig[InputP] = Field(default=convert_input_to_user_message)
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] = Field(default=return_last_message)


    # response_format: Optional[Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]] = None
    # return_on: Union[Literal["content", "tool_call", "message"], ToolName, Tool, list[ToolName | Tool]] = "content"

    # @property
    # def input_type(self) -> Type[InputT]:
    #     return self.input_parser.__annotations__.get("input", Message)
    
    @property
    def output_type(self) -> Type[OutputT]:
        if isinstance(self.output_parser, OutputParserConfig):
            return self.output_parser.output_type
        elif isinstance(self.output_parser, type):
            return self.output_parser
        else:
            self.output_parser.__annotations__.get("output", Message)
    
    @property
    def return_type(self) -> Type[ReturnT]:
        if isinstance(self.output_parser, OutputParserConfig):
            return self.output_parser.return_type
        elif isinstance(self.output_parser, type):
            return self.output_parser
        else:
            self.output_parser.__annotations__.get("return", Message)





class FunctionConfig(ChatConfig, _FunctionConfig[InputP, InputReturnT, OutputT, ReturnT], Generic[InputP, InputReturnT, OutputT, ReturnT]):
    component_type: ClassVar = "function"
    input_parser: Callable[InputP, InputReturnT | Callable[..., InputReturnT]] | InputParserConfig[InputP, InputReturnT] | RAGConfig[InputP] = Field(default=convert_input_to_user_message)
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] = Field(default=return_last_message)
    exception_handlers: Optional[dict[Type[Exception], Callable[..., ReturnT]]] = None

# class FunctionConfig(ChatConfig, _FunctionConfig[InputT, OutputT, ReturnT], Generic[InputT, OutputT, ReturnT]):
#     component_type: ClassVar = "function"
#     input_parser: Callable[[InputT], dict|str] = Field(default=convert_input_to_user_message)
#     output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] = Field(default=return_last_message)
#     exception_handlers: Optional[dict[Type[Exception], Callable[..., ReturnT]]] = None


# FunctionConfig()
from typing import TYPE_CHECKING, Generic, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, Concatenate

from ..types.annotations import InputP, InputReturnT, OutputT, ReturnT, NewInputP, NewInputReturnT, NewOutputT, NewReturnT
from ..types import (
    Message,
    Tool,
    ToolCall,
    Field,
)


from ..components.prompt_templates import PromptTemplate
from ._base_configs import BaseConfig
from .chat_config import ChatConfig
from .llm_config import LLMConfig
from .rag_config import RAGConfig
from .tool_caller_config import ToolCallerConfig
from .input_parser_config import InputParserConfig
from .output_parser_config import OutputParserConfig
from .rag_config import RAGConfig

if TYPE_CHECKING:
    from ..components.chats import Chat


def convert_input_to_user_message(input: Message|str) -> Message:
    if isinstance(input, Message):
        return input if input.role == "user" else Message(content=input.content, role="user")
    return Message(content=input)
    
def return_last_message(output: "Chat") -> Message:
    return output.last_message

class _FunctionConfig(BaseConfig, Generic[InputP, InputReturnT, OutputT, ReturnT]):
    component_type: ClassVar = "function"
    provider: ClassVar[str] = "default" 

    stateless: bool = True
    input_parser: Callable[InputP, InputReturnT] | Callable[InputP, Callable[..., InputReturnT]] | InputParserConfig[InputP, InputReturnT] | RAGConfig[InputP] | "_FunctionConfig[InputP, Any, Any, InputReturnT]  | _FunctionConfig"
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] | "_FunctionConfig[..., Any, Any, ReturnT] | _FunctionConfig" = Field(default=return_last_message)
    
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


class FunctionConfig(ChatConfig[InputP], _FunctionConfig[InputP, InputReturnT, OutputT, ReturnT], Generic[InputP, InputReturnT, OutputT, ReturnT]):
    component_type: ClassVar = "function"
    provider: ClassVar[str] = "default"

    input_parser: Callable[InputP, InputReturnT] | Callable[InputP, Callable[..., InputReturnT]] | InputParserConfig[InputP, InputReturnT] | RAGConfig[InputP] | _FunctionConfig[InputP, Any, Any, InputReturnT]  | _FunctionConfig = Field(default=convert_input_to_user_message)
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] | _FunctionConfig[..., Any, Any, ReturnT] | _FunctionConfig = Field(default=return_last_message)
    exception_handlers: Optional[dict[Type[Exception], Callable[..., ReturnT]]] = None
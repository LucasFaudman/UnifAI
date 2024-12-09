from typing import TYPE_CHECKING, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, ResponseFormatInput
from ..types import (
    Message,
    Tool,
    ToolCall,
)


from ..components.prompt_template import PromptTemplate
from ._base_configs import BaseConfig
from .chat_config import ChatConfig
from .llm_config import LLMConfig
from .rag_config import RAGConfig
from .tool_caller_config import ToolCallerConfig
from .output_parser_config import OutputParserConfig, OutputT, ReturnT

from typing import TypeAlias, TypeVar, Generic
if TYPE_CHECKING:
    from ..components.chat import Chat

from pydantic import BaseModel, Field
InputT = TypeVar('InputT', bound=BaseModel) # Input type of the function

class _FunctionConfig(BaseConfig, Generic[InputT, OutputT, ReturnT]):
    component_type: ClassVar = "function"
    provider: ClassVar[str] = "default" 

    stateless: bool = True
    
    llm: Optional[LLMConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    llm_model: Optional[ModelName] = None

    system_prompt: Optional[str | PromptTemplate | Callable[...,str]] = None
    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None
    
    input_parser: Callable[[InputT], dict|str] = lambda x: x.model_dump() if isinstance(x, BaseModel) else x
    prompt_template: str | PromptTemplate | Callable[..., str] = "{input}"
    rag_config: Optional[RAGConfig | ComponentName] = None
    
    response_format: Optional[Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]] = None
    return_on: Union[Literal["content", "tool_call", "message"], ToolName, Tool, list[ToolName | Tool]] = "content"
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT]

    @property
    def input_type(self) -> Type[InputT]:
        return self.input_parser.__annotations__.get("input", Message)
    
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



def get_message_content(input: Message) -> str:
    return input.content or ""

def return_as_message(output: Message) -> Message:
    return output

class FunctionConfig(ChatConfig, _FunctionConfig[InputT, OutputT, ReturnT], Generic[InputT, OutputT, ReturnT]):
    input_parser: Callable[[InputT], dict|str] = Field(default=get_message_content)
    output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] = Field(default=return_as_message)
    exception_handlers: Optional[dict[Type[Exception], Callable[..., ReturnT]]] = None


FunctionConfig()
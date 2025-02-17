from typing import TYPE_CHECKING, Any, Type, TypeVar, Generic, Optional
from functools import partial
from json import loads, JSONDecodeError

from .._base_components._base_output_parser import OutputParser, OutputParserConfig, OutputT, ReturnT
from ...exceptions import OutputParserError
from ...types import Message, ToolCall
from ...utils.typing_utils import is_base_model

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from ..chats import Chat

T = TypeVar("T", bound=BaseModel)

def pydantic_parse_one(output: "Chat|dict|ToolCall|str|Message|None", model: Type[T]|T) -> T:
    if isinstance(model, BaseModel):
        model = model.__class__
    try:
        if isinstance(output, dict):
            return model.model_validate(output)
        if isinstance(output, ToolCall):
            return model.model_validate(output.arguments)
        if last_message := getattr(output, "last_message", None):
            output = last_message
        if isinstance(output, Message):
            if output.tool_calls:
                return model.model_validate(output.tool_calls[0].arguments, strict=False)
            else:
                output = output.content       
        if output:
            return model.model_validate_json(output)
    except ValidationError as e:
        raise OutputParserError(
            message=f"Error validating output as {model.__class__.__name__} output: {e}",
            original_exception=e
        )
    raise OutputParserError(message=f"Error No output to parse as {model.__class__.__name__} output: {output}")
    

def pydantic_parse_many(outputs: "list[Chat|dict|ToolCall|str|Message|None]", model: Type[T]|T) -> list[T]:
    return [pydantic_parse_one(output, model) for output in outputs]

def pydantic_parse(output: "Chat|dict|ToolCall|str|Message|None|list[dict|ToolCall|str|Message]", model: Type[T]|T) -> T|list[T]: 
    if isinstance(output, list):
        return pydantic_parse_many(output, model)
    return pydantic_parse_one(output, model)


ModelT = TypeVar('ModelT', bound=BaseModel)
class PydanticParser(OutputParser[OutputT, ModelT], Generic[OutputT, ModelT]):
    provider = "pydantic"
    config_class: Type[OutputParserConfig[OutputT, ModelT]] = OutputParserConfig

    def __init__(
            self, 
            model: Optional[Type[ModelT]] = None, 
            output_type: Type[OutputT] = Message, 
            **init_kwargs
        ) -> None:
        config: OutputParserConfig[OutputT, ModelT] = (
            init_kwargs.pop('config', None)
            or self.config_class(
                provider=self.provider, 
                output_type=output_type, 
                return_type=model,
            )
        )
        super().__init__(config, **init_kwargs)

    def _setup(self) -> None:
        super()._setup()
        if not is_base_model(self.return_type):
            raise ValueError(f"return_type/model must be a subclass of pydantic.BaseModel. Got: {self.return_type}")
        self.model = self.return_type
        if not self._callable:         
            self._callable = partial(pydantic_parse, model=self.model)
        
    @classmethod
    def from_model(cls, model: Type[ModelT], output_type: Type[OutputT] = Message) -> "PydanticParser[OutputT, ModelT]":
        return cls(model=model, output_type=output_type)

class PydanticMessageParser(PydanticParser[Message, ModelT]):
    def __init__(self, model: Type[ModelT], **init_kwargs) -> None:
        super().__init__(model=model, output_type=Message, **init_kwargs)

class PydanticToolCallParser(PydanticParser[ToolCall, ModelT]):
    def __init__(self, model: Type[ModelT], **init_kwargs) -> None:
        super().__init__(model=model, output_type=ToolCall, **init_kwargs)


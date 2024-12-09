from typing import TYPE_CHECKING, Any, Type, TypeVar, Generic
from json import loads, JSONDecodeError

from ...exceptions import OutputParserError
from ...types import Message, ToolCall

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from ..chat import Chat

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
                return model.model_validate(output.tool_calls[0].arguments)
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
OutputT = TypeVar('OutputT', "Chat", Message, ToolCall) # Type of the output of the function to be parsed by the output_parser
class PydanticParser(Generic[ModelT, OutputT]):
    def __init__(self, model: Type[ModelT], output_type: Type[OutputT] = Message) -> None:
        self.model = model
        self.output_type = output_type

    def __call__(self, output: OutputT) -> ModelT:
        return pydantic_parse_one(output, self.model)
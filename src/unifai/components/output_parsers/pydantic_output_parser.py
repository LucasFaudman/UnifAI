from typing import Any, Type, TypeVar
from json import loads, JSONDecodeError

from ...exceptions import OutputParserError
from ...types import Message, ToolCall
from ...client.chat import Chat
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def pydantic_parse_one(output: Chat|dict|ToolCall|str|Message|None, model: Type[T]|T) -> T:
    if isinstance(model, BaseModel):
        model = model.__class__
    try:
        if isinstance(output, dict):
            return model.model_validate(output)
        if isinstance(output, ToolCall):
            return model.model_validate(output.arguments)
        if isinstance(output, Chat):
            output = output.last_message        
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
    

def pydantic_parse_many(outputs: list[Chat|dict|ToolCall|str|Message|None], model: Type[T]|T) -> list[T]:
    return [pydantic_parse_one(output, model) for output in outputs]


def pydantic_parse(output: Chat|dict|ToolCall|str|Message|None|list[dict|ToolCall|str|Message], model: Type[T]|T) -> T|list[T]: 
    if isinstance(output, list):
        return pydantic_parse_many(output, model)
    return pydantic_parse_one(output, model)
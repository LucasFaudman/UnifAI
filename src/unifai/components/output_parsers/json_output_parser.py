from typing import Any
from json import loads, JSONDecodeError

from ...exceptions import OutputParserError
from ...types import Message


def json_parse_one(output: str|Message|None) -> Any:
    if isinstance(output, Message):
        output = output.content
    if output is None:
        return None
    try:
        return loads(output)
    except JSONDecodeError as e:
        raise OutputParserError(message=f"Error parsing JSON output: {output}", original_exception=e)

def json_parse_many(outputs: list[str|Message|None]) -> list[Any]:
    return [json_parse_one(output) for output in outputs]

def json_parse(output: str|Message|None|list[str|Message|None]) -> Any|list[Any]:
    if isinstance(output, list):
        return json_parse_many(output)
    return json_parse_one(output)
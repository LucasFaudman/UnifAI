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
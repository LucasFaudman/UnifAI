from typing import Any, TypeVar, Generic, Type, Optional
from json import loads, JSONDecodeError

from ...exceptions import OutputParserError
from ...types import Message, ToolCall
from .._base_components._base_output_parser import OutputParser, OutputParserConfig, OutputT, ReturnT

JSONReturnT = TypeVar('JSONReturnT', dict, list, str, int, float, bool, None)

def json_parse_one(output: str|Message|ToolCall|None) -> JSONReturnT:
    if isinstance(output, Message):
        output = output.content
    if isinstance(output, ToolCall):
        return output.args
    if output is None:
        return None
    try:
        return loads(output)
    except JSONDecodeError as e:
        raise OutputParserError(message=f"Error parsing JSON output: {output}", original_exception=e)

def json_parse_many(output: list[str|Message|ToolCall|None]) -> JSONReturnT:
    return list(map(json_parse_one, output))

def json_parse(output: str|Message|ToolCall|None|list[str|Message|ToolCall|None]) -> JSONReturnT:
    if isinstance(output, list):
        return json_parse_many(output)
    return json_parse_one(output)


class JSONParser(OutputParser[OutputT, JSONReturnT], Generic[OutputT, JSONReturnT]):
    provider = "json"
    config_class: Type[OutputParserConfig[OutputT, JSONReturnT]] = OutputParserConfig

    def __init__(self, config: OutputParserConfig[OutputT, JSONReturnT], **init_kwargs) -> None:                
        super().__init__(config, **init_kwargs)

    def _parse_output(self, output: OutputT) -> JSONReturnT:
        if not isinstance((_parsed := json_parse(output)), self.return_type):
            raise OutputParserError(f"Error parsing output as {self.return_type}. Got type {type(_parsed)}: {_parsed=}")
        return _parsed

class JSONMessageParser(JSONParser[Message, JSONReturnT], Generic[JSONReturnT]):
    def __init__(self, return_type: Type[JSONReturnT], **init_kwargs) -> None:
        config = init_kwargs.pop('config', None) or OutputParserConfig[Message, JSONReturnT](provider=self.provider, output_type=Message, return_type=return_type)
        super().__init__(config, **init_kwargs)

class JSONMessage2DictParser(JSONMessageParser[dict]):
    def __init__(self, return_type: type[dict] = dict, **init_kwargs) -> None:
        super().__init__(return_type, **init_kwargs)

class JSONMessage2ListParser(JSONMessageParser[list]):
    def __init__(self, return_type: type[list] = list, **init_kwargs) -> None:
        super().__init__(return_type, **init_kwargs)
    
class JSONToolCallParser(JSONParser[ToolCall, JSONReturnT], Generic[JSONReturnT]):
    def __init__(self, return_type: Type[JSONReturnT], **init_kwargs) -> None:
        config = init_kwargs.pop('config', None) or OutputParserConfig[ToolCall, JSONReturnT](provider=self.provider, output_type=ToolCall, return_type=return_type)
        super().__init__(config, **init_kwargs)

class JSONToolCall2ListParser(JSONToolCallParser[list]):
    def __init__(self, return_type: type[list] = list, **init_kwargs) -> None:
        super().__init__(return_type, **init_kwargs)

class JSONToolCall2DictParser(JSONToolCallParser[dict]):
    def __init__(self, return_type: type[dict] = dict, **init_kwargs) -> None:
        super().__init__(return_type, **init_kwargs)







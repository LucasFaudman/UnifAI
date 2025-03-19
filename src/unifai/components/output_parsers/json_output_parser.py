from typing import Any, TypeVar, Generic, Type, Optional
from json import loads, JSONDecodeError
from functools import partial

from ...exceptions import OutputParserError
from ...utils.typing_utils import is_type_and_subclass
from ...types import Message, ToolCall
from .._base_components._base_output_parser import OutputParser, OutputParserConfig, OutputT, ReturnT

JSONReturnT = TypeVar('JSONReturnT', dict, list, str, int, float, bool, None)

def json_parse_one(
        output: str|Message|ToolCall|None, 
        return_type: Optional[Type[JSONReturnT]] = None,
    ) -> JSONReturnT:
    if isinstance(output, Message):
        output = output.content
    if output is None:
        _parsed = None        
    elif isinstance(output, ToolCall):
        _parsed = output.arguments
    else:
        try:
            _parsed = loads(output)
        except JSONDecodeError as e:
            raise OutputParserError(message=f"Error parsing JSON output: {output}", original_exception=e)
    if return_type and not isinstance(_parsed, return_type):
        raise OutputParserError(f"Error parsing output as {return_type}. Got type {type(_parsed)}: {_parsed=}")
    return _parsed

def json_parse_many(
        output: list[str|Message|ToolCall|None],
        return_type: Optional[Type[JSONReturnT]] = None,
        ) -> list[JSONReturnT]:
    return [json_parse_one(o, return_type) for o in output]

def json_parse(
        output: str|Message|ToolCall|None|list[str|Message|ToolCall|None],
        return_type: Optional[Type[JSONReturnT]] = None,
        ) -> JSONReturnT | list[JSONReturnT]:
    if isinstance(output, list):
        return json_parse_many(output, return_type)
    return json_parse_one(output, return_type)


class JSONParser(OutputParser[OutputT, JSONReturnT], Generic[OutputT, JSONReturnT]):
    provider = "json"
    config_class: Type[OutputParserConfig[OutputT, JSONReturnT]] = OutputParserConfig

    def _setup(self) -> None:
        super()._setup()
        if self.return_type not in (dict, list, str, int, float, bool, None):
            raise ValueError(f"return_type must be one of (dict, list, str, int, float, bool) or None. Got: {self.return_type}")
        if not self._callable:
            self._callable = partial(json_parse, return_type=self.return_type)

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







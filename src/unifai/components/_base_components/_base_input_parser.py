from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Generic, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self
from abc import abstractmethod

from .__base_component import UnifAIComponent

from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, ProviderName, GetResult, QueryResult, RerankedQueryResult
from ...exceptions import UnifAIError, OutputParserError
from ...configs.input_parser_config import InputParserConfig, InputP, InputReturnT

T = TypeVar("T")

class InputParser(UnifAIComponent[InputParserConfig[InputP, InputReturnT]], Generic[InputP, InputReturnT]):
    component_type = "input_parser"
    provider = "base"    
    config_class = InputParserConfig
    can_get_components = False

    def __init__(self, config: InputParserConfig[InputP, InputReturnT], **init_kwargs) -> None:
        super().__init__(config, **init_kwargs)
        self.input_parser = config.input_parser

    def _convert_exception(self, exception: Exception) -> UnifAIError:
        return OutputParserError(f"Error parsing input: {exception}", original_exception=exception)        

    @abstractmethod
    def _parse_input(self, *args: InputP.args, **kwargs: InputP.kwargs) -> InputReturnT:
        ...

    def parse_input(self, *args: InputP.args, **kwargs: InputP.kwargs) -> InputReturnT:
        return self._run_func(self._parse_input, input)
    
    def __call__(self, *args: InputP.args, **kwargs: InputP.kwargs) -> InputReturnT:
        return self.parse_input(*args, **kwargs)
    


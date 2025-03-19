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

    def _setup(self) -> None:
        super()._setup()
        if not self.config.callable and self._parse_input is InputParser._parse_input:
            raise ValueError("A callable must be passed in the InputParserConfig or the _parse_input method must be overridden in a subclass")
        self._callable = self.config.callable
        self.return_type = self.config.return_type

    def _convert_exception(self, exception: Exception) -> UnifAIError:
        return OutputParserError(f"Error parsing input: {exception}", original_exception=exception)        

    def _parse_input(self, *args: InputP.args, **kwargs: InputP.kwargs) -> InputReturnT:
        if not self._callable:
            # Should never happen, but just in case
            raise ValueError("A callable must be passed in the InputParserConfig or the _parse_input method must be overridden in a subclass")
        if callable(_parsed := self._callable(*args, **kwargs)):
            return _parsed(*args, **kwargs)
        return _parsed

    def parse_input(self, *args: InputP.args, **kwargs: InputP.kwargs) -> InputReturnT:
        return self._run_func(self._parse_input, *args, **kwargs)
    
    def __call__(self, *args: InputP.args, **kwargs: InputP.kwargs) -> InputReturnT:
        return self.parse_input(*args, **kwargs)
    
    @classmethod
    def from_callable(cls, func: Callable[InputP, InputReturnT]) -> "InputParser[InputP, InputReturnT]":
        return cls(config=cls.config_class(callable=func))
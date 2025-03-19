from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Generic, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self
from abc import abstractmethod

from .__base_component import UnifAIComponent

from ...type_conversions import tool_from_func
from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, ProviderName, GetResult, QueryResult, RerankedQueryResult
from ...exceptions import UnifAIError, OutputParserError
from ...configs.output_parser_config import OutputParserConfig, ReturnT, OutputT

T = TypeVar("T")

class OutputParser(UnifAIComponent[OutputParserConfig[OutputT, ReturnT]], Generic[OutputT, ReturnT]):
    component_type = "output_parser"
    provider = "base"    
    config_class = OutputParserConfig
    can_get_components = False

    def _setup(self) -> None:
        super()._setup()
        if not self.config.callable and self._parse_output is OutputParser._parse_output:
            raise ValueError("A callable must be passed in the OutputParserConfig or the _parse_output method must be overridden in a subclass")
        self._callable = self.config.callable
        self.output_type = self.config.output_type
        self.return_type = self.config.return_type

    def _convert_exception(self, exception: Exception) -> UnifAIError:
        return OutputParserError(f"Error parsing output: {exception}", original_exception=exception)        

    def _parse_output(self, output: OutputT) -> ReturnT:
        if not self._callable:
            # Should never happen, but just in case
            raise ValueError("A callable must be passed in the OutputParserConfig or the _parse_output method must be overridden in a subclass")
        return self._callable(output)

    def parse_output(self, output: OutputT) -> ReturnT:
        return self._run_func(self._parse_output, output)
    
    def __call__(self, output: OutputT) -> ReturnT:
        return self.parse_output(output)
    
    @classmethod
    def from_callable(cls, func: Callable[[OutputT], ReturnT]) -> "OutputParser[OutputT, ReturnT]":
        return cls(config=cls.config_class(callable=func))




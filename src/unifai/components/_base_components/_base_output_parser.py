from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Generic, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self
from abc import abstractmethod

from .__base_component import UnifAIComponent

from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, ProviderName, GetResult, QueryResult, RerankedQueryResult
from ...exceptions import UnifAIError, OutputParserError
from ...configs.output_parser_config import OutputParserConfig, ReturnT, OutputT

T = TypeVar("T")

class OutputParser(UnifAIComponent[OutputParserConfig[OutputT, ReturnT]], Generic[OutputT, ReturnT]):
    component_type = "output_parser"
    provider = "base"    
    config_class = OutputParserConfig
    can_get_components = False

    def __init__(self, config: OutputParserConfig[OutputT, ReturnT], **init_kwargs) -> None:
        super().__init__(config, **init_kwargs)
        self.output_type = config.output_type
        self.return_type = config.return_type

    def _convert_exception(self, exception: Exception) -> UnifAIError:
        return OutputParserError(f"Error parsing output: {exception}", original_exception=exception)        

    @abstractmethod
    def _parse_output(self, output: OutputT) -> ReturnT:
        ...

    def parse_output(self, output: OutputT) -> ReturnT:
        return self._run_func(self._parse_output, output)
    
    def __call__(self, output: OutputT) -> ReturnT:
        return self.parse_output(output)
    


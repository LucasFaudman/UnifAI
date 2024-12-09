from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator, Generic

from ..components.prompt_template import PromptTemplate
from ..components.output_parsers.json_output_parser import json_parse
from ..components.output_parsers.pydantic_output_parser import pydantic_parse
from ..components.tool_callers import ToolCaller

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput
from ..types import Message, MessageChunk, Tool, ToolCall, ToolInput, ToolChoiceInput
from ..type_conversions import tool_from_model
from ..utils import stringify_content

from ._base_components._base_component import UnifAIComponent
from .chat import BaseChat
from .ragpipe import RAGPipe
from .output_parsers.pydantic_output_parser import PydanticParser

from ..configs.rag_config import RAGConfig
from ..configs.function_config import FunctionConfig, InputT, OutputT, ReturnT

from pydantic import BaseModel, Field, ConfigDict

def is_base_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)

def is_tool_or_model(value: Any) -> bool:
    return isinstance(value, Tool) or is_base_model_type(value)

class Function(BaseChat[FunctionConfig[InputT, OutputT, ReturnT]], Generic[InputT, OutputT, ReturnT]):
    component_type = "function"
    provider = "base"
    config_class = FunctionConfig

    def __init__(self, config: FunctionConfig[InputT, OutputT, ReturnT], **init_kwargs) -> None:
        super().__init__(config, **init_kwargs)

    def reset(self) -> Self:
        self.clear_messages()
        # self._fully_initialized = False
        return self

    def _init_config_components(self) -> None:
        super()._init_config_components()
        self.output_parser = self.config.output_parser

    @property
    def output_parser(self) -> Callable[[OutputT], ReturnT]:
        return self._output_parser

    @output_parser.setter
    def output_parser(self, output_parser: Type[ReturnT] | Callable[[OutputT], ReturnT]):
        if isinstance(output_parser, (Tool, PydanticParser)) or isinstance(output_parser, type) and issubclass(output_parser, BaseModel):  
            if isinstance(output_parser, Tool):
                output_parser = PydanticParser.from_model(output_parser.callable)
                output_tool = output_parser
            elif isinstance(output_parser, PydanticParser):
                output_tool = tool_from_model(output_parser.model)
            else:
                output_parser, output_tool = PydanticParser.from_model(output_parser), tool_from_model(output_parser)
            
            if self.tools is None:
                self.tools = [output_tool]
            if output_tool.name not in self.tools:
                self.add_tool(output_tool)
            if self._tool_choice_queue is None:
                self.tool_choice = output_tool.name
            elif self._tool_choice_queue[-1] != output_tool.name:
                self._tool_choice_queue.append(output_tool.name)
            self.return_on = output_tool.name
        
        self._output_parser = output_parser
          
    def _get_ragpipe(
            self, 
            provider_config_or_name: "ProviderName | RAGConfig | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "RAGPipe":
        return self._get_component("ragpipe", provider_config_or_name, **init_kwargs)
        
    @property
    def ragpipe(self) -> "Optional[RAGPipe]":
        if (ragpipe := getattr(self, "_ragpipe", None)) is not None:
            return ragpipe
        if self.config.rag_config:
            self._ragpipe = self._get_ragpipe(self.config.rag_config)
            return self._ragpipe
        return None
    
    @ragpipe.setter
    def ragpipe(self, value: "Optional[RAGPipe | ProviderName | RAGConfig | tuple[ProviderName, ComponentName]]"):
        if value is None or isinstance(value, RAGPipe):
            self._ragpipe = value
        else:
            self._ragpipe = self._get_ragpipe(value)

    
    def handle_exception(self, exception: Exception) -> ReturnT:
        handlers = self.config.exception_handlers
        if not handlers:
            raise exception
        
        if not (handler := handlers.get(exception.__class__)):
            for error_type, handler in handlers.items():
                if isinstance(exception, error_type):
                    return handler(self, exception) 
        raise exception
        
    def prepare_message(self, *input: InputT|str, **kwargs) -> Message:
        if len(input) > 1:
            raise ValueError("Only one input argument is allowed")
        
        input_str = None
        if input:
            _input = input[0]
            if isinstance(_input, str):
                input_str = _input
            else:
                parsed_input = self.config.input_parser(_input)
                if isinstance(parsed_input, str):
                    input_str = parsed_input
                else:
                    kwargs.update(parsed_input)
        
        prompt_template = self.config.prompt_template
        if kwargs:
            if input_str:
                kwargs["input"] = input_str
            if isinstance((prompt_template), (PromptTemplate, str)):
                input_str = prompt_template.format(**kwargs)
            else:
                input_str = prompt_template(**kwargs)
        elif input_str is None:
                input_str = prompt_template if isinstance(prompt_template, str) else prompt_template()
        
        if self.ragpipe:
            prompt = self.ragpipe.prompt(query=input_str)
        else:
            prompt = input_str
                
        return Message(role="user", content=prompt)
    
    def _get_output(self) -> OutputT:
        output_type = self.config.output_type
        if output_type is str:
            return self.last_content
        elif output_type is Message:
            return self.last_message
        elif output_type is ToolCall:
            return self.last_tool_call
        else:
            return self

    def parse_output(self) -> ReturnT:
        output = self._get_output()
        return self.output_parser(output)

    def parse_output_stream(self) -> Generator[MessageChunk, None, ReturnT]:
        output = self._get_output()
        for output_parser in self.output_parsers:
            if isinstance(output_parser, Function):
                output = yield from output_parser.stream(output)
            else:
                output = output_parser(output)
        return output  

    def __call__(self, *input: InputT|str, **kwargs) -> ReturnT:
        try:
            message = self.prepare_message(*input, **kwargs)
            self.send_message(message)
            return self.parse_output()
        except Exception as error:
            return self.handle_exception(error)
        finally:
            if self.config.stateless:
                self.reset()

    def stream(self, *input: InputT|str, **kwargs) -> Generator[MessageChunk, None, ReturnT]:
        try:
            message = self.prepare_message(input=input, **kwargs)
            yield from self.send_message_stream(message)
            output = yield from self.parse_output_stream()
            return output
        except Exception as error:
            return self.handle_exception(error)
        finally:
            if self.config.stateless:
                self.reset()
                    
            
    # Aliases so func()==func.exec() and func.stream()==func.exec_stream()
    exec = __call__
    exec_stream = stream
    
    # def __or__(self, other: Union['UnifAIFunction', Callable]) -> 'UnifAIFunction':
    #     """Implements the | operator by appending to output_parsers"""
    #     if other is self:
    #         raise ValueError("Cannot pipe a function into itself")        
    #     new_func = self.with_config()  # Create a copy
    #     new_func.output_parsers.append(other)
    #     return new_func
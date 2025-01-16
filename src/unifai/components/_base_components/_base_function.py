from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, TypeVar, Union, Self, Iterable, Mapping, Generator, Generic, ParamSpec, cast

from ._base_prompt_template import PromptTemplate


from ...types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput
from ...types import Message, MessageChunk, Tool, ToolCall, ToolInput, ToolChoiceInput
from ...type_conversions import tool_from_model
from ...utils import stringify_content

from ._base_chat import BaseChat
from ..ragpipes import RAGPipe
from ..input_parsers import InputParser
from ..output_parsers import OutputParser
from ..output_parsers.pydantic_output_parser import PydanticParser

from ...configs.rag_config import RAGConfig
from ...configs.function_config import _FunctionConfig, FunctionConfig, InputParserConfig, OutputParserConfig, InputP, InputReturnT, OutputT, ReturnT

from pydantic import BaseModel, Field, ConfigDict

FunctionConfigT = TypeVar('FunctionConfigT', bound=FunctionConfig)
NewInputP = ParamSpec('NewInputP')
NewInputReturnT = TypeVar('NewInputReturnT', Message, str, Message | str)
NewOutputT = TypeVar('NewOutputT')
NewReturnT = TypeVar('NewReturnT')

class BaseFunction(BaseChat[FunctionConfigT], Generic[FunctionConfigT, InputP, InputReturnT, OutputT, ReturnT]):
    component_type = "function"
    provider = "base"
    config_class: Type[FunctionConfigT]

    def _setup(self) -> None:
        super()._setup()
        # self._get_input_parser: Callable[[ProviderName | InputParserConfig[InputP, InputReturnT] | tuple[ProviderName, ComponentName]], InputParser[InputP, InputReturnT]] = self.init_kwargs.pop("_get_input_parser")
        # self._get_output_parser: Callable[[ProviderName | OutputParserConfig[OutputT, ReturnT] | tuple[ProviderName, ComponentName]], OutputParser[OutputT, ReturnT]] = self.init_kwargs.pop("_get_output_parser")
        # self._get_ragpipe: Callable[[ProviderName | RAGConfig[InputP] | tuple[ProviderName, ComponentName]], RAGPipe[InputP]] = self.init_kwargs.pop("_get_ragpipe")
        # self._get_function: Callable[[ProviderName | FunctionConfig[InputP, Any, Any, InputReturnT] | tuple[ProviderName, ComponentName]], "BaseFunction[FunctionConfig[InputP, Any, Any, InputReturnT], InputP, Any, Any, InputReturnT]"] = self.init_kwargs.pop("_get_function")
        self._get_input_parser: Callable[..., InputParser[InputP, InputReturnT]] = self.init_kwargs.pop("_get_input_parser")
        self._get_output_parser: Callable[..., OutputParser[OutputT, ReturnT]] = self.init_kwargs.pop("_get_output_parser")
        self._get_ragpipe: Callable[..., RAGPipe[InputP]] = self.init_kwargs.pop("_get_ragpipe")
        self._get_function: Callable[..., "BaseFunction[FunctionConfig[InputP, Any, Any, InputReturnT], InputP, Any, Any, InputReturnT]"] = self.init_kwargs.pop("_get_function")
        self._set_input_parser(self.config.input_parser)

    def _init_config_components(self) -> None:
        super()._init_config_components()
        self._set_output_parser(self.config.output_parser)        

    def reset(self) -> Self:
        self.clear_messages()
        # self._fully_initialized = False
        return self

    @property
    def input_parser(self) -> Callable[InputP, InputReturnT | Callable[..., InputReturnT]]:
        return self._input_parser
    
    @input_parser.setter
    def input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[NewInputP] | 
                    FunctionConfig[NewInputP, Any, Any, NewInputReturnT]) -> None:
        self.set_input_parser(input_parser)

    def _set_input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[NewInputP] | 
                    _FunctionConfig[NewInputP, Any, Any, NewInputReturnT]) -> None:
        
        if callable(input_parser):
            self._input_parser = input_parser
        elif isinstance(input_parser, InputParserConfig):
            self._input_parser = self._get_input_parser(input_parser)
        elif isinstance(input_parser, RAGConfig):
            self._input_parser = self._get_ragpipe(input_parser)
        elif isinstance(input_parser, FunctionConfig):
            self._input_parser = self._get_function(input_parser)
        else:
            raise ValueError(f"Invalid input_parser: {input_parser}")
        return
        
    def set_input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[NewInputP] | 
                    FunctionConfig[NewInputP, Any, Any, NewInputReturnT]):
        self._set_input_parser(input_parser)
        self = cast("BaseFunction[FunctionConfig[NewInputP, NewInputReturnT, OutputT, ReturnT], NewInputP, NewInputReturnT, OutputT, ReturnT]", self)
        return self

    @property
    def output_parser(self) -> Callable[[OutputT], ReturnT]:
        return self._output_parser
    
    @output_parser.setter
    def output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     FunctionConfig[..., Any, Any, NewReturnT]) -> None:
        self.set_output_parser(output_parser)
        
    def _set_output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     _FunctionConfig[..., Any, Any, NewReturnT]) -> None:

        if isinstance(output_parser, OutputParserConfig):
            output_parser = self._get_output_parser(output_parser)
        elif isinstance(output_parser, FunctionConfig):
            output_parser = self._get_function(output_parser)

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
        return

    def set_output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     FunctionConfig[..., Any, Any, NewReturnT]):
        
        self._set_output_parser(output_parser)
        self = cast("BaseFunction[FunctionConfig[InputP, InputReturnT, NewOutputT, NewReturnT], InputP, InputReturnT, NewOutputT, NewReturnT]", self)
        return self
          
    
    def handle_exception(self, exception: Exception) -> ReturnT:
        handlers = self.config.exception_handlers
        if not handlers:
            raise exception
        
        if not (handler := handlers.get(exception.__class__)):
            for error_type, handler in handlers.items():
                if isinstance(exception, error_type):
                    return handler(self, exception) 
        raise exception
        
    def prepare_message(self, *args: InputP.args, **kwargs: InputP.kwargs) -> Message:
        _input = self.input_parser(*args, **kwargs)
        if callable(_input):
            _input = _input(*args, **kwargs)
        if isinstance(_input, Message):
            return _input
        if not isinstance(_input, str):
            _input = stringify_content(_input)
        return Message(role="user", content=_input)
            
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
        if isinstance(self.output_parser, BaseFunction):
            output = yield from self.output_parser.stream(output)
        else:
            output = self.output_parser(output)
        return output

    def __call__(self, *args: InputP.args, **kwargs: InputP.kwargs) -> ReturnT:
        try:
            message = self.prepare_message(*args, **kwargs)
            self.send_message(message)
            return self.parse_output()
        except Exception as error:
            return self.handle_exception(error)
        finally:
            if self.config.stateless:
                self.reset()

    def stream(self, *args: InputP.args, **kwargs: InputP.kwargs) -> Generator[MessageChunk, None, ReturnT]:
        try:
            message = self.prepare_message(*args, **kwargs)
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
    
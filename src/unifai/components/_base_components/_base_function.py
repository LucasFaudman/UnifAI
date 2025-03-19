from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, TypeVar, Union, Self, Iterable, Mapping, Generator, Generic, ParamSpec, cast, TextIO, Pattern
from re import compile as re_compile
from ._base_prompt_template import PromptModel


from ...types.annotations import InputP, InputReturnT, OutputT, ReturnT, NewInputP, NewInputReturnT, NewOutputT, NewReturnT
from ...types import Message, MessageChunk, Tool, ToolCall, ToolInput, ToolChoiceInput
from ...type_conversions import tool_from_model, tool_from_func
from ...utils import stringify_content
from ...utils.typing_utils import is_base_model

from ._base_chat import BaseChat
from ..ragpipes import RAGPipe
from ..input_parsers import InputParser
from ..output_parsers import OutputParser
from ..output_parsers.pydantic_output_parser import PydanticParser

from ...configs._base_configs import ComponentConfig
from ...configs.rag_config import RAGConfig
from ...configs.function_config import _FunctionConfig, FunctionConfig, InputParserConfig, OutputParserConfig, InputP, InputReturnT, OutputT, ReturnT

from pydantic import BaseModel, Field, ConfigDict

FunctionConfigT = TypeVar('FunctionConfigT', bound=FunctionConfig)


class BaseFunction(BaseChat[FunctionConfigT], Generic[FunctionConfigT, InputP, InputReturnT, OutputT, ReturnT]):
    component_type = "function"
    provider = "base"
    config_class: Type[FunctionConfigT]

    def _setup(self) -> None:
        super()._setup()
        self._structured_outputs_mode = None
        self._output_tool = None
        self._input_parser = None
        self._output_parser = None
        self._set_structured_outputs_mode(self.config.structured_outputs_mode)
        self._set_input_parser(self.config.input_parser)

    def _init_config_components(self) -> None:
        super()._init_config_components()
        self._set_output_parser(self.config.output_parser)   

    def reset(self) -> Self:
        self.clear_messages()
        self._output_tool = None
        self._set_structured_outputs_mode(self.config.structured_outputs_mode)
        self._set_input_parser(self.config.input_parser)
        self._set_output_parser(self.config.output_parser)
        return self
    
    def handle_exception(self, exception: Exception) -> ReturnT:
        handlers = self.config.error_handlers
        if not handlers:
            raise exception
        
        if not (handler := handlers.get(exception.__class__)):
            for error_type, handler in handlers.items():
                if isinstance(exception, error_type):
                    return handler(self, exception) 
        raise exception    

    @property
    def input_parser(self) -> Callable[InputP, InputReturnT | Callable[..., InputReturnT]]:
        return self._input_parser
    
    @input_parser.setter
    def input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[..., NewInputP] | 
                    FunctionConfig[NewInputP, Any, Any, NewInputReturnT]) -> None:
        self._set_input_parser(input_parser)

    def _set_input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[..., NewInputP] | 
                    _FunctionConfig[NewInputP, Any, Any, NewInputReturnT]) -> None:
        if isinstance(input_parser, ComponentConfig):
            input_parser = self._get_component_with_config(
                input_parser, valid_component_types=("input_parser", "ragpipe", "function")
            )
        if not callable(input_parser):
            raise ValueError(f"Input parser must be callable, or an InputParserConfig, RAGConfig, or FunctionConfig. Got: {input_parser}")
        if not isinstance(input_parser, (BaseFunction, RAGPipe, InputParser, PromptModel)):
            input_parser = InputParser.from_callable(input_parser)
        self._input_parser = input_parser

        
    def set_input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[..., NewInputP] | 
                    FunctionConfig[NewInputP, Any, Any, NewInputReturnT]):
        self._set_input_parser(input_parser)
        return cast("BaseFunction[FunctionConfig[NewInputP, NewInputReturnT, OutputT, ReturnT], NewInputP, NewInputReturnT, OutputT, ReturnT]", self)


    @property
    def output_tool(self) -> Optional[Tool]:
        return self._output_tool
    
    @output_tool.setter
    def output_tool(self, output_tool: Optional[Tool]) -> None:
        self._set_output_tool(output_tool)

    def _set_output_tool(self, output_tool: Optional[Tool]) -> None:
        if self.structured_outputs_mode == "tool_call" or output_tool is None:
            if self._output_tool:
                # Remove old output tool before adding new one or setting to None
                self.pop_tool(self._output_tool)

            if output_tool:                        
                self.add_tool(output_tool)
                if not self._tool_choice_queue or self._tool_choice_queue[-1] != output_tool.name:
                    self.append_tool_choice(output_tool.name)
                self.return_on = output_tool.name
        elif self.structured_outputs_mode == "json_schema":
            self.response_format = output_tool
            self.return_on = "content"

        self._output_tool = output_tool
        return

    @property
    def output_parser(self) -> Callable[[OutputT], ReturnT]:
        return self._output_parser
    
    @output_parser.setter
    def output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     FunctionConfig[..., Any, Any, NewReturnT]) -> None:
        self._set_output_parser(output_parser)

    def _set_output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     _FunctionConfig[..., Any, Any, NewReturnT]) -> None:

        output_tool = None
        if isinstance(output_parser, ComponentConfig):
            output_parser = self._get_component_with_config(
                output_parser, valid_component_types=("output_parser", "function")
            )
        if not callable(output_parser):
            raise ValueError(f"Output parser must be callable, an OutputParserConfig, or a FunctionConfig. Got: {output_parser}")

        # if isinstance(output_parser, BaseFunction):
            # pass
            # output_tool = tool_from_func(output_parser.input_parser)
        if isinstance(output_parser, PydanticParser):
            output_tool = tool_from_model(output_parser.model)
        elif isinstance(output_parser, Tool):
            if not is_base_model(output_parser.callable):
                raise ValueError(f"Tool callable must be a pydantic model when used as an output parser Got: {output_parser.callable}")
            output_tool = output_parser
            output_parser = PydanticParser.from_model(output_parser.callable)
        elif is_base_model(output_parser):
            output_tool = tool_from_model(output_parser)
            output_parser = PydanticParser.from_model(output_parser)
        elif not isinstance(output_parser, (OutputParser, BaseFunction)):
            # output_tool = tool_from_func(output_parser)
            output_parser = OutputParser.from_callable(output_parser)

        self.output_tool = output_tool        
        self._output_parser = output_parser
        return


    def set_output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     FunctionConfig[..., Any, Any, NewReturnT]):
        
        self._set_output_parser(output_parser)
        return cast("BaseFunction[FunctionConfig[InputP, InputReturnT, NewOutputT, NewReturnT], InputP, InputReturnT, NewOutputT, NewReturnT]", self)
              

    @property
    def structured_outputs_mode(self) -> Literal["tool_call", "json_schema"]:
        return self._structured_outputs_mode
        
    @structured_outputs_mode.setter
    def structured_outputs_mode(self, mode: Literal["tool_call", "json_schema"]) -> None:
        self._set_structured_outputs_mode(mode)

    def _set_structured_outputs_mode(self, mode: Literal["tool_call", "json_schema"]) -> None:
        if mode not in ("tool_call", "json_schema"):
            raise ValueError(f"Mode must be one of ['tool_call', 'json_schema']. Got: {mode}")
        self._structured_outputs_mode = mode
        if self._output_tool:
            self._set_output_tool(self._output_tool)
        return
        
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
        if (last_message := self.last_message) is None:
            raise ValueError("No last message found. parse_output and parse_output_stream can only be called after running the function")

        output_type: Type[OutputT] = getattr(self.output_parser, "output_type", Message) # type: ignore
        if isinstance(last_message, output_type):
            return last_message
        if isinstance(last_message.content, output_type):
            return last_message.content
        if isinstance(self.last_tool_call, output_type):
            return self.last_tool_call
        if last_message.tool_calls and isinstance(last_message.tool_calls[0], output_type):
            return last_message.tool_calls[0]
        if isinstance(self, output_type):
            return self
        
        raise ValueError(f"No valid output of type {output_type} found in last message. Last message: {last_message}")

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

    def print_stream(
            self, 
            end: Optional[str] = "",
            file: Optional[TextIO] = None,
            flush: bool = True, 
            replacements: Optional[dict[str|Pattern, str]] = None,
            *args: InputP.args, 
            **kwargs: InputP.kwargs
            ) -> ReturnT:
        
        bufsize = 0
        buffer = []
        _replacements = {}
        
        if replacements:
            for old, new in replacements.items():
                if isinstance(old, str):
                    old = re_compile(old)
                _replacements[old] = new
                bufsize = max(bufsize, len(old.pattern))

        def _replace(joined: str) -> str:
            if not _replacements:
                return joined
            for old, new in _replacements.items():
                joined = old.sub(new, joined)    
            return joined
            
        stream_generator = self.stream(*args, **kwargs)
        _current_bufsize = 0
        try:            
            while True:                
                if (chunk := next(stream_generator)) and (content := chunk.content):
                    buffer.append(content)
                    _current_bufsize += len(content)
                    if _current_bufsize >= bufsize:
                        print(_replace(''.join(buffer)), end=end, file=file, flush=flush)
                        buffer = []
                        _current_bufsize = 0
        except StopIteration as stop:
            output = stop.value
        if buffer:
            print(_replace(''.join(buffer)), end=end, file=file, flush=flush)
        return output
    
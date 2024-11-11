from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from ..components.prompt_template import PromptTemplate
from ..components.output_parsers.json_output_parser import json_parse
from ..components.output_parsers.pydantic_output_parser import pydantic_parse
from ..components.tool_callers import ToolCaller

from ..types import Message, MessageChunk, Tool, ToolInput, ToolChoiceInput
from ..type_conversions import stringify_content, tool_from_model

from .chat import Chat
from .rag_engine import RAGEngine, RAGConfig

from pydantic import BaseModel, Field, ConfigDict

def is_base_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)

def is_tool_or_model(value: Any) -> bool:
    return isinstance(value, Tool) or is_base_model_type(value)

class FunctionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    stateless: bool = True

    provider: Optional[str] = None           
    model: Optional[str] = None

    prompt_template: PromptTemplate|str = PromptTemplate("{content}", value_formatters={Message: lambda m: m.content})
    prompt_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    rag_config: Optional[RAGConfig|str] = None
    
    system_prompt: Optional[str|PromptTemplate|Callable[...,str]] = None
    system_prompt_kwargs: dict[str, Any] = Field(default_factory=dict)

    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None   
    response_format: Optional[Literal["text", "json"]|Type[BaseModel]|Tool|dict[Literal["json_schema"], dict|Type[BaseModel]|Tool]] = None
    return_on: Union[Literal["content", "tool_call", "message"], str, Tool, list[str|Tool]] = "content"
    return_as: Literal["self", 
                       "messages", 
                       "last_message", 
                       "last_content",
                       "last_tool_call",
                       "last_tool_call_args",
                       "last_tool_calls", 
                       "last_tool_calls_args"
                       ] = "self"
    output_parser: Optional[Callable|Type[BaseModel]|BaseModel] = None
    output_parser_kwargs: dict[str, Any] = Field(default_factory=dict)

    tools: Optional[list[ToolInput]] = None            
    tool_choice: Optional[ToolChoiceInput] = None
    enforce_tool_choice: bool = True
    tool_choice_error_retries: int = 3
    tool_callables: Optional[dict[str, Callable]] = None
    tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller
    tool_caller_kwargs: dict[str, Any] = Field(default_factory=dict)

    max_messages_per_run: int = 10
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop_sequences: Optional[list[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    exception_handlers: Optional[Mapping[Type[Exception], Callable[..., Any]]] = None


class UnifAIFunction(Chat):
    def __init__(
            self, 
            config: FunctionConfig,
            rag_engine: Optional[RAGEngine] = None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.output_parsers = []

        if output_parser := config.output_parser:
            self.output_parsers.append(output_parser)

        if response_format := config.response_format: 
            if response_format == "json":
                self.output_parsers.append(json_parse)
            elif is_tool_or_model(response_format):
                self.output_parsers.append(response_format)
                if not isinstance(response_format, Tool):
                    response_format = tool_from_model(response_format)
                config.return_on = response_format.name
                config.response_format = None

                if config.tools is None:
                    config.tools = []
                if response_format not in config.tools:
                    config.tools.append(response_format)

                if (tool_choice := config.tool_choice) is None:
                    config.tool_choice = response_format
                elif isinstance(tool_choice, (Tool, str)) and tool_choice != response_format and tool_choice != response_format.name:
                    config.tool_choice = [tool_choice, response_format]
                elif isinstance(tool_choice, list) and response_format not in tool_choice:
                    config.tool_choice.append(response_format)

            # elif isinstance(response_format, dict):
            #     schema = response_format.get("json_schema")
            #     if isinstance(schema, dict):
            #         self.output_parsers.append(json_parse_one)
            #     elif is_tool_or_model(schema):
            #         self.output_parsers.append(schema)

        self.config = config
        self.rag_engine = rag_engine
        self.reset()
        

    def reset(self) -> Self:
        self.clear_messages()
        
        config = self.config
        if config.provider:
            self.set_provider(config.provider, config.model)
        elif config.model:
            self.set_model(config.model)
        
        system_prompt_kwargs = config.system_prompt_kwargs
        if (isinstance((system_prompt := config.system_prompt), PromptTemplate)
            or (isinstance(system_prompt, str) and system_prompt_kwargs)
            ):
            system_prompt = system_prompt.format(**system_prompt_kwargs)
        elif callable(system_prompt):
            system_prompt = system_prompt(**system_prompt_kwargs)
        else:
            system_prompt = system_prompt # None or (str with no kwargs to format)
        
        example_messages = []
        if examples := config.examples:
            for example in examples:
                if isinstance(example, Message):
                    example_messages.append(example)
                else:
                    example_messages.append(Message(role="user", content=stringify_content(example['input'])))
                    example_messages.append(Message(role="assistant", content=stringify_content(example['response'])))
                        
        self.set_messages(example_messages, system_prompt)
        self.return_on = config.return_on
        self.set_response_format(config.response_format)
        self.set_tools(config.tools)
        self.set_tool_choice(config.tool_choice)
        self.set_tool_caller(config.tool_caller_class_or_instance, config.tool_callables, config.tool_caller_kwargs)
        self.enforce_tool_choice = config.enforce_tool_choice
        self.tool_choice_error_retries = config.tool_choice_error_retries
        self.max_messages_per_run = config.max_messages_per_run
        self.max_tokens = config.max_tokens
        self.frequency_penalty = config.frequency_penalty
        self.presence_penalty = config.presence_penalty
        self.seed = config.seed
        self.stop_sequences = config.stop_sequences   
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p
        return self


    def update_config(self, **kwargs) -> Self:
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        return self
    
    
    def with_config(self, **kwargs) -> "UnifAIFunction":
        return self.copy(
            config=self.config.model_copy(update=kwargs), 
            rag_engine=self.rag_engine
        )

    
    def handle_exception(self, exception: Exception):
        handlers = self.config.exception_handlers
        if not handlers:
            raise exception
        
        if not (handler := handlers.get(exception.__class__)):
            for error_type, handler in handlers.items():
                if isinstance(exception, error_type):
                    break
        if not handler or not handler(self, exception):
            raise exception
        

    def prepare_input(self, *args, **kwargs) -> Any:
        if args:
            if len(args) != 1:
                raise ValueError("Only one positional argument is allowed, the value for '{content}' if present in the prompt template, got: ", args)
            kwargs["content"] = args[0]
        
        config = self.config
        prompt = self.config.prompt_template.format(**{**config.prompt_template_kwargs, **kwargs})
        if self.rag_engine:
            prompt = self.rag_engine.ragify(query=prompt)
        return prompt
    

    def parse_output(self, *args, **kwargs):
        config = self.config
        output = getattr(self, config.return_as) if config.return_as != "self" else self
        for output_parser in self.output_parsers:
            if isinstance(output_parser, Tool) and output_parser.callable:
                output_parser = output_parser.callable

            # TODO: Multiple possible output parsers based on if content, tool_name, message, etc.
            if is_base_model_type(output_parser):
                output = pydantic_parse(output, model=output_parser, **config.output_parser_kwargs)
            else:
                output = output_parser(output, **config.output_parser_kwargs)
        return output               


    def __call__(self, *args, **kwargs) -> Any:
        try:
            rag_prompt = self.prepare_input(*args, **kwargs)
            self.send_message(rag_prompt)
            return self.parse_output()
        except Exception as error:
            self.handle_exception(error)
        finally:
            if self.config.stateless:
                self.reset()
    

    def stream(
            self,
            *args,        
            **kwargs,
        ) -> Generator[MessageChunk, None, Any]:
        try:
            rag_prompt = self.prepare_input(*args, **kwargs)
            yield from self.send_message_stream(rag_prompt)
            return self.parse_output()
        except Exception as error:
            self.handle_exception(error)
        finally:
            if self.config.stateless:
                self.reset()
                    

    # Aliases so func()==func.exec() and func.stream()==func.exec_stream()
    exec = __call__
    exec_stream = stream
    


from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator
from ..components.prompt_template import PromptTemplate
from ..components.output_parsers import pydantic_parse, json_parse_one
from ..exceptions import UnifAIError
from ..types import Message, MessageChunk, Tool
from ..type_conversions import stringify_content, tool_from_model
from .chat import Chat
from .specs import FuncSpec
from .rag_engine import RAGEngine

from pydantic import BaseModel

def is_tool_or_model(value: Any) -> bool:
    return isinstance(value, Tool) or (isinstance(value, type) and issubclass(value, BaseModel))

class AIFunction(Chat):
    def __init__(
            self, 
            spec: FuncSpec,
            rag_engine: Optional[RAGEngine] = None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.output_parsers = []

        if response_format := spec.response_format: 
            if response_format == "json":
                self.output_parsers.append(json_parse_one)
            elif is_tool_or_model(response_format):
                self.output_parsers.append(response_format)
                if not isinstance(response_format, Tool):
                    response_format = tool_from_model(response_format)
                spec.return_on = response_format.name
                spec.response_format = None

                if spec.tools is None:
                    spec.tools = []
                if response_format not in spec.tools:
                    spec.tools.append(response_format)

                if (tool_choice := spec.tool_choice) is None:
                    spec.tool_choice = response_format
                elif isinstance(tool_choice, (Tool, str)) and tool_choice != response_format and tool_choice != response_format.name:
                    spec.tool_choice = [tool_choice, response_format]
                elif isinstance(tool_choice, list) and response_format not in tool_choice:
                    spec.tool_choice.append(response_format)

            # elif isinstance(response_format, dict):
            #     schema = response_format.get("json_schema")
            #     if isinstance(schema, dict):
            #         self.output_parsers.append(json_parse_one)
            #     elif is_tool_or_model(schema):
            #         self.output_parsers.append(schema)

        if output_parser := spec.output_parser:
            self.output_parsers.append(output_parser)

        self.spec = spec
        self.rag_engine = rag_engine
        self.reset()
        

    @property
    def name(self) -> str:
        return self.spec.name


    def reset(self) -> Self:
        self.clear_messages()
        
        spec = self.spec
        if spec.provider:
            self.set_provider(spec.provider, spec.model)
        elif spec.model:
            self.set_model(spec.model)
        
        system_prompt_kwargs = spec.system_prompt_kwargs
        if (isinstance((system_prompt := spec.system_prompt), PromptTemplate)
            or (isinstance(system_prompt, str) and system_prompt_kwargs)
            ):
            system_prompt = system_prompt.format(**system_prompt_kwargs)
        elif callable(system_prompt):
            system_prompt = system_prompt(**system_prompt_kwargs)
        else:
            system_prompt = system_prompt # None or (str with no kwargs to format)
        
        example_messages = []
        if examples := spec.examples:
            for example in examples:
                if isinstance(example, Message):
                    example_messages.append(example)
                else:
                    example_messages.append(Message(role="user", content=stringify_content(example['input'])))
                    example_messages.append(Message(role="assistant", content=stringify_content(example['response'])))
                        
        self.set_messages(example_messages, system_prompt)
        self.return_on = spec.return_on
        self.set_response_format(spec.response_format)
        self.set_tools(spec.tools)
        self.set_tool_choice(spec.tool_choice)
        self.enforce_tool_choice = spec.enforce_tool_choice
        self.tool_choice_error_retries = spec.tool_choice_error_retries
        self.max_messages_per_run = spec.max_messages_per_run
        self.max_tokens = spec.max_tokens
        self.frequency_penalty = spec.frequency_penalty
        self.presence_penalty = spec.presence_penalty
        self.seed = spec.seed
        self.stop_sequences = spec.stop_sequences   
        self.temperature = spec.temperature
        self.top_k = spec.top_k
        self.top_p = spec.top_p
        return self


    def update_spec(self, **kwargs) -> Self:
        for key, value in kwargs.items():
            setattr(self.spec, key, value)
        return self
    
    
    def with_spec(self, **kwargs) -> "AIFunction":
        return self.copy(
            spec=self.spec.model_copy(update=kwargs), 
            rag_engine=self.rag_engine
        )

    
    def handle_error(self, error: UnifAIError):
        error_handlers = self.spec.error_handlers
        if not error_handlers:
            raise error
        
        if not (handler := error_handlers.get(error.__class__)):
            for error_type, handler in error_handlers.items():
                if isinstance(error, error_type):
                    break
        if not handler or not handler(self, error):
            raise error
        

    def prepare_input(self, *args, **kwargs) -> Any:
        if args:
            if len(args) != 1:
                raise ValueError("Only one positional argument is allowed, the value for '{content}' if present in the prompt template, got: ", args)
            kwargs["content"] = args[0]
        
        spec = self.spec
        prompt = self.spec.prompt_template.format(**{**spec.prompt_template_kwargs, **kwargs})
        if self.rag_engine:
            prompt = self.rag_engine.ragify(query=prompt)
        return prompt
    

    def parse_output(self, *args, **kwargs):
        spec = self.spec
        output = getattr(self, spec.return_as) if spec.return_as != "self" else self
        for output_parser in self.output_parsers:
            if isinstance(output_parser, Tool) and output_parser.callable:
                output_parser = output_parser.callable

            # TODO: Multiple possible output parsers based on if content, tool_name, message, etc.
            if isinstance(output_parser, BaseModel) or (isinstance(output_parser, type) and issubclass(output_parser, BaseModel)):
                output = pydantic_parse(output, output_parser, **spec.output_parser_kwargs)
            else:
                output = output_parser(output, **spec.output_parser_kwargs)
        return output               


    def __call__(self, *args, **kwargs) -> Any:
        try:
            rag_prompt = self.prepare_input(*args, **kwargs)
            self.send_message(rag_prompt)
            return self.parse_output()
        except UnifAIError as error:
            self.handle_error(error)
        finally:
            if self.spec.reset_on_return:
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
        except UnifAIError as error:
            self.handle_error(error)
        finally:
            if self.spec.reset_on_return:
                self.reset()
                    

    # Aliases so func()==func.exec() and func.stream()==func.exec_stream()
    exec = __call__
    exec_stream = stream
    


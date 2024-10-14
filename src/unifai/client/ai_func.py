from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from .chat import Chat
from .specs import FuncSpec
from .rag_engine import RAGEngine
from ..components.prompt_template import PromptTemplate
from ..components.output_parser import pydantic_parse
from ..exceptions import UnifAIError
from ..types import Message, MessageChunk
from ..type_conversions import make_few_shot_prompt, stringify_content
from pydantic import BaseModel


class AIFunction(Chat):
    def __init__(
            self, 
            spec: FuncSpec,
            rag_engine: Optional[RAGEngine] = None,
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.spec = spec
        self.rag_engine = rag_engine
        self.reset()
        

    def reset(self) -> Self:
        self.clear_messages()
        spec = self.spec

        if spec.provider:
            self.set_provider(spec.provider, spec.model)
        elif spec.model:
            self.set_model(spec.model)
        
        if isinstance((system_prompt := spec.system_prompt), PromptTemplate):
            system_prompt = system_prompt.format(**spec.system_prompt_kwargs)
        elif callable(system_prompt):
            system_prompt = system_prompt(**spec.system_prompt_kwargs)
        else:
            system_prompt = system_prompt # None or str
        
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
        return self.copy(spec=self.spec.model_copy(update=kwargs), rag_engine=self.rag_engine)

    
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
            rag_spec = spec.rag_spec
            prompt = self.rag_engine.ragify(
                query=prompt,
                retreiver_kwargs=rag_spec.retreiver_kwargs if rag_spec else None, # To all standalone use with RAGEngine no RAGSpec
                reranker_kwargs=rag_spec.reranker_kwargs if rag_spec else None,
            )        
        return prompt
    

    def parse_output(self, *args, **kwargs):
        spec = self.spec
        output = getattr(self, spec.return_as) if spec.return_as != "self" else self
        if output_parser := spec.output_parser:
            # TODO: Multiple possible output parsers based on if content, tool_name, message, etc.
            if isinstance(output_parser, BaseModel) or (isinstance(output_parser, type) and issubclass(output_parser, BaseModel)):
                output = pydantic_parse(output, output_parser, **spec.output_parser_kwargs)
            else:
                output = output_parser(output, **spec.output_parser_kwargs)
        return output               


    def exec(
            self,
            *args,        
            **kwargs,
        ) -> Any:
        rag_prompt = self.prepare_input(*args, **kwargs)
        try:
            self.send_message(rag_prompt)
        except UnifAIError as error:
            self.handle_error(error)
        output = self.parse_output()
        if self.spec.reset_on_return:
            self.reset()
        return output
    
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.exec(*args, **kwargs)    


    def exec_stream(
            self,
            *args,        
            **kwargs,
        ) -> Generator[MessageChunk, None, Any]:
        rag_prompt = self.prepare_input(*args, **kwargs)        
        try:
           yield from self.send_message_stream(rag_prompt)
        except UnifAIError as error:
            self.handle_error(error)
        output = self.parse_output()
        if self.spec.reset_on_return:
            self.reset()
        return output        
    


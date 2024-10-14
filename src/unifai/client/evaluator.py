from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from .chat import Chat
from .specs import EvalSpec
from .rag_engine import RAGEngine
from ..components.prompt_template import PromptTemplate
from ..components.output_parser import pydantic_parse
from ..exceptions import UnifAIError
from ..types import Message, MessageChunk
from ..type_conversions import make_few_shot_prompt, stringify_content
from pydantic import BaseModel

class AIEvaluator:
    def __init__(
            self, 
            spec: EvalSpec,
            chat: Chat,
            rag_engine: Optional[RAGEngine] = None,
            ):
        
        self.spec = spec.model_copy()
        self.chat = chat
        self.rag_engine = rag_engine
        self.reset()


    # def set_chat(self, chat: Chat):
    #     self.chat = chat

    # def set_spec(self, spec: EvalSpec) -> Self:
    #     self.spec = spec.model_copy()
    #     return self

    def update_spec(self, **kwargs) -> Self:
        for key, value in kwargs.items():
            setattr(self.spec, key, value)
        return self
    
    def with_spec(self, **kwargs) -> "AIEvaluator":
        return AIEvaluator(
            spec=self.spec, 
            chat=self.chat.copy(), 
            rag_engine=self.rag_engine
        ).update_spec(**kwargs)

    
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
        

    def reset(self) -> Self:
        chat = self.chat
        chat.clear_messages()
        spec = self.spec

        if spec.provider:
            chat.set_provider(spec.provider, spec.model)
        elif spec.model:
            chat.set_model(spec.model)
        
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
                        
        chat.set_messages(example_messages, system_prompt)
        chat.return_on = spec.return_on
        chat.set_response_format(spec.response_format)
        chat.set_tools(spec.tools)
        chat.set_tool_choice(spec.tool_choice)
        chat.enforce_tool_choice = spec.enforce_tool_choice
        chat.tool_choice_error_retries = spec.tool_choice_error_retries
        chat.max_tokens = spec.max_tokens
        chat.frequency_penalty = spec.frequency_penalty
        chat.presence_penalty = spec.presence_penalty
        chat.seed = spec.seed
        chat.stop_sequences = spec.stop_sequences   
        chat.temperature = spec.temperature
        chat.top_k = spec.top_k
        chat.top_p = spec.top_p
        return self


    def prepare_input(self, *args, **kwargs) -> Any:
        if args and kwargs:
            raise ValueError("Cannot provide both args and kwargs")
        if args and len(args) > 1:
            raise ValueError("Only one positional argument is allowed")
        if not args and not kwargs:
            raise ValueError("Must provide either args or kwargs")
        if args:
            kwargs["content"] = args[0]
        
        spec = self.spec
        prompt = self.spec.prompt_template.format(**{**spec.prompt_template_kwargs, **kwargs})
        if self.rag_engine and (rag_spec := spec.rag_spec):
            prompt = self.rag_engine.ragify(
                query=prompt, 
                retreiver_kwargs=rag_spec.retreiver_kwargs,
                reranker_kwargs=rag_spec.reranker_kwargs,
            )
        
        return prompt
    

    def parse_output(self, *args, **kwargs):
        spec = self.spec
        output = getattr(self.chat, spec.return_as) if spec.return_as != "chat" else self.chat
        if output_parser := spec.output_parser:
            if isinstance(output_parser, BaseModel) or (isinstance(output_parser, type) and issubclass(output_parser, BaseModel)):
                output = pydantic_parse(output, output_parser, **spec.output_parser_kwargs)
            else:
                output = output_parser(output, **spec.output_parser_kwargs)
        return output               


    def run(
            self,
            *args,        
            **kwargs,
        ) -> Any:
        rag_prompt = self.prepare_input(*args, **kwargs)
        try:
            self.chat.send_message(rag_prompt)
        except UnifAIError as error:
            self.handle_error(error)
        output = self.parse_output()
        if self.spec.reset_on_return:
            self.reset()
        return output
    

    def run_stream(
            self,
            *args,        
            **kwargs,
        ) -> Generator[MessageChunk, None, Any]:
        rag_prompt = self.prepare_input(*args, **kwargs)        
        try:
           yield from self.chat.send_message_stream(rag_prompt)
        except UnifAIError as error:
            self.handle_error(error)
        output = self.parse_output()
        if self.spec.reset_on_return:
            self.reset()
        return output        
    

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)
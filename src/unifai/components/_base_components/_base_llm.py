from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Callable, Iterator, Iterable, Generator
from abc import abstractmethod

from ._base_adapter import UnifAIAdapter
from .__base_component import convert_exceptions, convert_exceptions_generator


from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embeddings, Usage
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError
from ...utils import update_kwargs_with_locals, combine_dicts

from ...configs import LLMConfig

T = TypeVar("T")

class LLM(UnifAIAdapter[LLMConfig]):
    component_type = "llm"
    provider = "base"    
    config_class = LLMConfig
    can_get_components = True
    default_llm_model = "default"
    
    _system_prompt_input_type: Literal["first_message", "kwarg"] = "first_message"

    # Abstract Methods
    @abstractmethod
    def _list_models(self) -> list[str]:
        ...

    @abstractmethod
    def _get_chat_response(
            self,
            stream: bool,
            messages: list[Any],
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,  
            tools: Optional[list[dict]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            response_format: Optional[str] = None,            
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None, 
            **kwargs
            ) -> Any:
        ...

    # Convert Objects from UnifAI to LLM format
        # Messages
    @abstractmethod
    def format_user_message(self, message: Message) -> Any:
        ...

    @abstractmethod
    def format_assistant_message(self, message: Message) -> Any:
        ...    
        
    @abstractmethod        
    def format_tool_message(self, message: Message) -> Any:
        ...
    
    def split_tool_message(self, message: Message) -> Iterator[Message]:     
        # Not abstract but here since common to subclass but can default to just yielding as is
        yield message

    @abstractmethod
    def format_system_message(self, message: Message) -> Any:
        ...
    
        # Images
    @abstractmethod        
    def format_image(self, image: Image) -> Any:
        ...    
    
        # Tools        
    @abstractmethod        
    def format_tool(self, tool: Tool) -> Any:
        ...
        
    @abstractmethod        
    def format_tool_choice(self, tool_choice: str) -> Any:
        ...    

        # Response Format
    @abstractmethod        
    def format_response_format(self, response_format: Union[str, dict]) -> Any:
        ...


    # Convert Objects from LLM to UnifAI format    
        # Images
    @abstractmethod        
    def parse_image(self, response_image: Any, **kwargs) -> Image:
        ...

        # Tool Calls
    @abstractmethod        
    def parse_tool_call(self, response_tool_call: Any, **kwargs) -> ToolCall:
        ...
    
        # Response Info (Model, Usage, Done Reason, etc.)
    @abstractmethod        
    def parse_done_reason(self, response_obj: Any, **kwargs) -> str|None:
        ...
    
    @abstractmethod    
    def parse_usage(self, response_obj: Any, **kwargs) -> Usage|None:
        ...
    
    @abstractmethod    
    def parse_response_info(self, response: Any, **kwargs) -> ResponseInfo:
        ...
    
        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    @abstractmethod        
    def parse_message(self, response: Any, **kwargs) -> tuple[Message, Any]:
        ...     
    
    @abstractmethod    
    def parse_stream(self, response: Any, **kwargs) -> Generator[MessageChunk, None, tuple[Message, Any]]:
        ...


    # Concrete Methods
    def list_models(self) -> list[str]:
        return self._run_func(self._list_models)

    @property
    def default_model(self) -> str:
        return self.config.default_model or self.default_llm_model

    def chat(
            self,
            messages: list[T],     
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,  
            tools: Optional[list[dict]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            response_format: Optional[str] = None,            
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None, 
            **kwargs
            ) -> tuple[Message, T]:
        kwargs["stream"] = False
        update_kwargs_with_locals(kwargs, locals())
        response = self._run_func(self._get_chat_response, **kwargs)
        unifai_message, client_message = self.parse_message(response, **kwargs)
        return unifai_message, client_message
    
    def chat_stream(
            self,
            messages: list[T],     
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,  
            tools: Optional[list[dict]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            response_format: Optional[str] = None,            
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None, 
            **kwargs
            ) -> Generator[MessageChunk, None, tuple[Message, T]]:
        kwargs["stream"] = True        
        update_kwargs_with_locals(kwargs, locals())
        response = self._run_func(self._get_chat_response, **kwargs)
        unifai_message, client_message = yield from self._run_generator(self.parse_stream, response, **kwargs)
        return unifai_message, client_message

    def format_message(self, message: Message) -> Any:
        if message.role == "user":
            return self.format_user_message(message)
        elif message.role == "assistant":
            return self.format_assistant_message(message)
        elif message.role == "tool":
            return self.format_tool_message(message)
        elif message.role == "system":
            return self.format_system_message(message)        
        else:
            raise ValueError(f"Invalid message role: {message.role}")

    def format_messages(self, messages: list[Message]) -> Iterable:
        for message in messages:
            if message.role == "tool":
                yield from map(self.format_message, self.split_tool_message(message))
            else:
                yield self.format_message(message)
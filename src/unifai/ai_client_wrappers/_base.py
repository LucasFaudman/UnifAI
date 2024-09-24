from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator

from json import dumps as json_dumps

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, EmbedResult, Usage
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError

T = TypeVar("T")

class BaseAIClientWrapper:
    provider = "base"
    default_model = "mistral:7b-instruct"

    def import_client(self) -> Type:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def init_client(self, **client_kwargs) -> Any:
        self.client_kwargs.update(client_kwargs)
        self._client = self.import_client()(**self.client_kwargs)
        return self._client

    def __init__(self, **client_kwargs):
        self._client = None
        self.client_kwargs = client_kwargs

    @property
    def client(self) -> Type:
        if self._client is None:
            # return self.init_client(**self.client_kwargs)
            return self.run_func_convert_exceptions(self.init_client, **self.client_kwargs)
        return self._client
    

    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: Exception) -> UnifAIError:
        raise NotImplementedError("This method must be implemented by the subclass")


    def run_func_convert_exceptions(self, func: Callable[..., T], *args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise self.convert_exception(e) from e


    # Convert Objects from UnifAI to AI Provider format        
        # Messages
    def prep_input_message(self, message: Message) -> Any:
        if message.role == "user":
            return self.prep_input_user_message(message)
        elif message.role == "assistant":
            return self.prep_input_assistant_message(message)
        elif message.role == "tool":
            return self.prep_input_tool_message(message)
        elif message.role == "system":
            return self.prep_input_system_message(message)        
        else:
            raise ValueError(f"Invalid message role: {message.role}")
    
    def prep_input_user_message(self, message: Message) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def prep_input_assistant_message(self, message: Message) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")    
        
    def prep_input_tool_message(self, message: Message) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def split_tool_message(self, message: Message) -> Iterator[Message]:     
        yield message

    def prep_input_system_message(self, message: Message) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def prep_input_messages_and_system_prompt(self, 
                                              messages: list[Message], 
                                              system_prompt_arg: Optional[str] = None
                                              ) -> tuple[list, Optional[str]]:
        raise NotImplementedError("This method must be implemented by the subclass")
       
        # Images
    def prep_input_image(self, image: Image) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")    
    
        # Tools
    def prep_input_tool_call(self, tool_call: ToolCall) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
        
    def prep_input_tool(self, tool: Tool) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
        
    def prep_input_tool_choice(self, tool_choice: str) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")    

        # Response Format
    def prep_input_response_format(self, response_format: Union[str, dict]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")


    # Convert Objects from AI Provider to UnifAI format    
        # Images
    def extract_image(self, response_image: Any) -> Image:
        raise NotImplementedError("This method must be implemented by the subclass")

        # Tool Calls
    def extract_tool_call(self, response_tool_call: Any) -> ToolCall:
        raise NotImplementedError("This method must be implemented by the subclass")
    
        # Response Info (Model, Usage, Done Reason, etc.)    
    def extract_done_reason(self, response_obj: Any) -> str|None:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def extract_usage(self, response_obj: Any) -> Usage|None:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def extract_response_info(self, response: Any) -> ResponseInfo:
        raise NotImplementedError("This method must be implemented by the subclass")
    
        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def extract_assistant_message_both_formats(self, response: Any) -> tuple[Message, Any]:
        raise NotImplementedError("This method must be implemented by the subclass")     
    
    def extract_stream_chunks(self, response: Any) -> Generator[MessageChunk, None, tuple[Message, Any]]:
        raise NotImplementedError("This method must be implemented by the subclass")
    # def split_tool_outputs_into_messages(self, tool_calls: Sequence[ToolCall], content: Optional[str] = None) -> Iterator[Message]:
    #     raise NotImplementedError("This method must be implemented by the subclass")


    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")    


    # Chat
    def get_chat_response(
            self,
            messages: list[Any],     
            model: str = default_model,
            system_prompt: Optional[str] = None,                    
            tools: Optional[list[dict]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            response_format: Optional[str] = None,
            stream: bool = False,
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
        raise ProviderUnsupportedFeatureError(f"{self.provider} does not support chat")
            
    # def chat(
    #         self,
    #         messages: list[T],     
    #         model: Optional[str] = None,
    #         system_prompt: Optional[str] = None,                   
    #         tools: Optional[list[Any]] = None,
    #         tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,            
    #         response_format: Optional[Union[str, dict[str, str]]] = None,

    #         max_tokens: Optional[int] = None,
    #         frequency_penalty: Optional[float] = None,
    #         presence_penalty: Optional[float] = None,
    #         seed: Optional[int] = None,
    #         stop_sequences: Optional[list[str]] = None, 
    #         temperature: Optional[float] = None,
    #         top_k: Optional[int] = None,
    #         top_p: Optional[float] = None, 

    #         **kwargs
    #         ) -> tuple[Message, T]:
    #     raise ProviderUnsupportedFeatureError(f"{self.provider} does not support chat")

    def chat(
            self,
            messages: list[T],     
            # model: str = default_model,
            # system_prompt: Optional[str] = None,                    
            # tools: Optional[list[dict]] = None,
            # tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            # response_format: Optional[str] = None,
            # stream: bool = False,
            # max_tokens: Optional[int] = None,
            # frequency_penalty: Optional[float] = None,
            # presence_penalty: Optional[float] = None,
            # seed: Optional[int] = None,
            # stop_sequences: Optional[list[str]] = None, 
            # temperature: Optional[float] = None,
            # top_k: Optional[int] = None,
            # top_p: Optional[float] = None, 

            **kwargs
            ) -> tuple[Message, T]:
        
        response = self.get_chat_response(messages=messages, **kwargs)
        std_message, client_message = self.extract_assistant_message_both_formats(response)
        return std_message, client_message
    
    def chat_stream(
            self,
            messages: list[T],     
            # model: str = default_model,
            # system_prompt: Optional[str] = None,                    
            # tools: Optional[list[dict]] = None,
            # tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            # response_format: Optional[str] = None,
            # stream: bool = False,
            # max_tokens: Optional[int] = None,
            # frequency_penalty: Optional[float] = None,
            # presence_penalty: Optional[float] = None,
            # seed: Optional[int] = None,
            # stop_sequences: Optional[list[str]] = None, 
            # temperature: Optional[float] = None,
            # top_k: Optional[int] = None,
            # top_p: Optional[float] = None, 

            **kwargs
            ) -> Generator[MessageChunk, None, tuple[Message, T]]:
        
        response = self.get_chat_response(messages=messages, **kwargs)
        std_message, client_message = yield from self.extract_stream_chunks(response)
        return std_message, client_message

    # Embeddings
    def embed(
            self,            
            input: str | Sequence[str],
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            **kwargs
            ) -> EmbedResult:
        raise ProviderUnsupportedFeatureError(f"{self.provider} does not support embeddings")
    
    
    def create_assistant(self, **kwargs):
        raise ProviderUnsupportedFeatureError(f"{self.provider} does not support chat")
        
    def update_assistant(self, ass_id, **kwargs):
        raise ProviderUnsupportedFeatureError(f"{self.provider} does not support chat")
    

    def create_thread(self):
        raise ProviderUnsupportedFeatureError(f"{self.provider} does not support chat")
    
    def create_run(self):
        raise ProviderUnsupportedFeatureError(f"{self.provider} does not support chat")    
    


   
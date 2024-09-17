from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator

from json import dumps as json_dumps

from unifai.types import Message, Tool, ToolCall, Image, ResponseInfo
from unifai.exceptions import UnifAIError

T = TypeVar("T")

class BaseAIClientWrapper:
    default_model = "mistral:7b-instruct"

    def import_client(self) -> Type:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def init_client(self, **client_kwargs) -> Any:
        self._client = self.import_client()(**client_kwargs)
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
    def extract_response_info(self, response: Any) -> ResponseInfo:
        raise NotImplementedError("This method must be implemented by the subclass")
    
        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def extract_assistant_message_both_formats(self, response: Any) -> tuple[Message, Any]:
        raise NotImplementedError("This method must be implemented by the subclass")     
    

    def split_tool_outputs_into_messages(self, tool_calls: Sequence[ToolCall], content: Optional[str] = None) -> Iterator[Message]:
        raise NotImplementedError("This method must be implemented by the subclass")


    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")    


    # Chat
    def chat(
            self,
            messages: list[T],     
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,                   
            tools: Optional[list[Any]] = None,
            tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,            
            response_format: Optional[Union[str, dict[str, str]]] = None,

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
        raise NotImplementedError("This method must be implemented by the subclass")


    # Generate
    def generate(
            self,
            model: Optional[str] = None,
            prompt: Optional[str] = None,
            **kwargs
            ):
        raise NotImplementedError("This method must be implemented by the subclass")
    

    # Embeddings
    def embeddings(
            self,
            model: Optional[str] = None,
            texts: Optional[Sequence[str]] = None,
            **kwargs
            ):
        raise NotImplementedError("This method must be implemented by the subclass")
    


    def create_assistant(self, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")
        
    def update_assistant(self, ass_id, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def create_thread(self):
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def create_run(self):
        raise NotImplementedError("This method must be implemented by the subclass")    
    


   
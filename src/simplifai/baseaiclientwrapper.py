from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar

from json import dumps as json_dumps
from ._types import Message, Tool, ToolCall, Image

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
            return self.init_client(**self.client_kwargs)
        return self._client
    


    def prep_input_message(self, message: Message) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def prep_input_messages_and_system_prompt(self, 
                                              messages: list[Message], 
                                              system_prompt_arg: Optional[str] = None
                                              ) -> tuple[list, Optional[str]]:
        raise NotImplementedError("This method must be implemented by the subclass")    
    
    def prep_input_tool(self, tool: Tool) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
        
    def prep_input_tool_call(self, tool_call: ToolCall) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def split_tool_call_outputs_into_messages(self, tool_calls: list[ToolCall]) -> list[Message]:
        raise NotImplementedError("This method must be implemented by the subclass")


    def prep_input_message_tool_call(self, message: Message, input_object: T) -> T:
        raise NotImplementedError("This method must be implemented by the subclass")

    def prep_input_message_tool_call_response(self, message: Message, input_object: T) -> T:
        raise NotImplementedError("This method must be implemented by the subclass")
        
    
    def prep_input_tool_choice(self, tool_choice: str) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def prep_input_response_format(self, response_format: Union[str, dict]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def extract_output_tool_call(self, tool_call: Any) -> ToolCall:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def extract_std_and_client_messages(self, response: Any) -> tuple[Message, Any]:
        raise NotImplementedError("This method must be implemented by the subclass")   

    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")    


    # Chat 
    def chat(
            self,
            messages: list[Message],     
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,                   
            tools: Optional[list[Any]] = None,
            tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,            
            response_format: Optional[Union[str, dict[str, str]]] = None,
            **kwargs
            ) -> Message:
        raise NotImplementedError("This method must be implemented by the subclass")

    # Embeddings

    def embeddings(
            self,
            model: Optional[str] = None,
            texts: Optional[Sequence[str]] = None,
            **kwargs
            ):
        raise NotImplementedError("This method must be implemented by the subclass")
    
    # Generate

    def generate(
            self,
            model: Optional[str] = None,
            prompt: Optional[str] = None,
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
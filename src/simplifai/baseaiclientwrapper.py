from typing import Type, Optional, Sequence, Any, Union, Literal

from json import dumps as json_dumps
from ._types import Message, Tool, ToolCall, Image

class BaseAIClientWrapper:
    default_model = "mistral:7b-instruct"

    def recursively_make_serializeable(self, obj):
        """Recursively makes an object serializeable by converting it to a dict or list of dicts and converting all non-string values to strings."""
        serializeable_types = (str, int, float, bool, type(None))
        if isinstance(obj, serializeable_types):
            return obj
        if isinstance(obj, dict):
            return {k: self.recursively_make_serializeable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.recursively_make_serializeable(item) for item in obj]
        return str(obj)

    def format_content(self, content):
        """Formats content for use a message content. If content is not a string, it is converted to json_"""
        if not isinstance(content, str):
            content = self.recursively_make_serializeable(content)
            content = json_dumps(content, indent=0)

        return content    


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
    
    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def prep_input_message(self, message: Union[Message, dict[str, str], str]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def prep_input_tool(self, tool: Union[Tool, dict]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def prep_input_tool_choice(self, tool_choice: Union[Tool, str, dict, Literal["auto", "required", "none"]]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def prep_input_tool_call(self, tool_call: ToolCall) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def prep_input_response_format(self, response_format: Union[str, dict[str, str]]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def extract_output_tool_call(self, tool_call: Any) -> ToolCall:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    # def extract_output_message(self, response: Any) -> Message:
    #     raise NotImplementedError("This method must be implemented by the subclass")   
    
    def extract_std_and_client_messages(self, response: Any) -> tuple[Message, Any]:
        raise NotImplementedError("This method must be implemented by the subclass")        

    
    # Chat 
    def chat(
            self,
            messages: list[Message],     
            model: Optional[str] = None,                    
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
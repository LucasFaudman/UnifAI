from typing import Optional, Union, Sequence, Any, Literal, Callable, Mapping
from pydantic import BaseModel

from .message import Message
from .tool import Tool

class EvalSpec(BaseModel):
    eval_type: str
    system_prompt: str = "Your role is to evaluate the content using the provided tool(s)." 
    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None   
    tools: Optional[list[Union[Tool, str, Callable]]] = None
    tool_choice: Optional[Union[str, list[str]]] = None
    return_on: Optional[Union[Literal["content", "tool_call", "message"], str, list[str]]] = None
    return_as: Literal["chat", 
                       "messages", 
                       "last_message", 
                       "last_content",
                       "last_tool_call",
                       "last_tool_call_args",
                       "last_tool_calls", 
                       "last_tool_calls_args"
                       ] = "chat"

    
    response_format: Optional[Union[str, dict[str, str]]] = None
    enforce_tool_choice: bool = True
    tool_choice_error_retries: int = 3


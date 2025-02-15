from typing import TYPE_CHECKING, Optional, Any
from ._base import UnifAIError
if TYPE_CHECKING:
    from ..types import Tool, ToolCall, ToolChoice, ToolName, ToolInput

class ToolError(UnifAIError):
    """Raised when an error occurs with a tool or tool call"""

class ToolValidationError(ToolError):
    """Raised when a tool parameter is invalid"""
    def __init__(self, 
                 message: str, 
                 tool_input: "ToolInput",
                 original_exception: Optional[Exception] = None
                 ):
        self.tool_input = tool_input
        super().__init__(message, original_exception)

class ToolNotFoundError(ToolError):
    """Raised when a tool is not found"""
    def __init__(self, 
                 message: str, 
                 tool_name: "ToolName",
                 original_exception: Optional[Exception] = None
                 ):
        self.tool_name = tool_name
        super().__init__(message, original_exception)

class ToolCallError(ToolError):
    """Raised when an error occurs during a tool call"""
    def __init__(self, 
                 message: str, 
                 tool_call: Optional["ToolCall"] = None,
                 tool_calls: Optional[list["ToolCall"]] = None,
                 tool: Optional["Tool"] = None,
                 original_exception: Optional[Exception] = None
                 ):
        self.tool_call = tool_call
        self.tool_calls = tool_calls
        self.tool = tool
        super().__init__(message, original_exception)

class ToolCallArgumentValidationError(ToolCallError):
    """Raised when the arguments for a tool call are invalid"""

class ToolCallableNotFoundError(ToolCallError):
    """Raised when a callable is not found for a tool"""

class ToolCallExecutionError(ToolCallError):
    """Raised when an error occurs while executing a tool call. (Calling the Tool's callable with the ToolCall's arguments)"""

class ToolChoiceError(ToolCallError):
    """Raised when a tool parameter choice is not obeyed"""
    def __init__(self, 
                 message: str, 
                 tool_call: Optional["ToolCall"] = None,
                 tool_calls: Optional[list["ToolCall"]] = None,
                 tool: Optional["Tool"] = None,
                 tool_choice: Optional["ToolName"] = None,
                 original_exception: Optional[Exception] = None
                 ):
        self.tool_choice = tool_choice
        super().__init__(message, tool_call, tool_calls, tool, original_exception)

class ToolChoiceErrorRetriesExceeded(ToolChoiceError):
    """Raised when the maximum number of tool choice errors is exceeded"""




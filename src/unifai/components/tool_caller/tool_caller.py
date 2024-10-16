from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from ...types import Tool, ToolCall
from ...exceptions.tool_errors import ToolCallExecutionError, ToolCallableNotFoundError, ToolCallArgumentValidationError


class ToolCaller:

    def __init__(
            self,
            tool_callables: dict[str, Callable[..., Any]],
            tool_argument_validators: Optional[dict[str, Callable[..., Any]]] = None,
            tools: Optional[list[Tool]] = None,
            tool_execution_error_retries: int = 0,            
    ):
        self.tool_callables = {tool.name: tool.callable for tool in tools if tool.callable} if tools else {}
        self.tool_callables.update(tool_callables) # tool_callables dict takes precedence over tools list
        self.tool_argument_validators = tool_argument_validators if tool_argument_validators is not None else {}
        self.tool_execution_error_retries = tool_execution_error_retries

    def set_tool_callables(self, tool_callables: dict[str, Callable[..., Any]]):
        self.tool_callables = tool_callables

    def update_tool_callables(self, tool_callables: dict[str, Callable[..., Any]]):
        self.tool_callables.update(tool_callables)

    def set_tool_argument_validators(self, tool_argument_validators: dict[str, Callable[..., Any]]):
        self.tool_argument_validators = tool_argument_validators
    
    def update_tool_argument_validators(self, tool_argument_validators: dict[str, Callable[..., Any]]):
        self.tool_argument_validators.update(tool_argument_validators)

    def validate_arguments(self, tool_call: ToolCall) -> Any:
        tool_name = tool_call.tool_name
        if (tool_argument_validator := self.tool_argument_validators.get(tool_name)) is None:
            return tool_call.arguments
        try:
            return tool_argument_validator(tool_call.arguments)
        except Exception as e:
            if isinstance(e, ToolCallArgumentValidationError):
                raise e
            raise ToolCallArgumentValidationError(
                message=f"Invalid arguments for tool '{tool_call}'",
                tool_call=tool_call,
                original_exception=e,
            )
        
    def call_tool(self, tool_call: ToolCall) -> ToolCall:
        tool_name = tool_call.tool_name
        
        if (tool_callable := self.tool_callables.get(tool_name)) is None:
            raise ToolCallableNotFoundError(
                message=f"Tool '{tool_name}' callable not found",
                tool_call=tool_call,
            )
        
        execution_retries = 0
        while execution_retries <= self.tool_execution_error_retries:
            try:
                tool_call.output = tool_callable(**tool_call.arguments)
                break
            except Exception as e:
                execution_retries += 1
                if execution_retries >= self.tool_execution_error_retries:
                    raise ToolCallExecutionError(
                        message=f"Error executing tool '{tool_name}'",
                        tool_call=tool_call,
                        original_exception=e,
                    )
        return tool_call


    def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        # Validate all arguments before calling any tools
        for tool_call in tool_calls:
            tool_call.arguments = self.validate_arguments(tool_call) # Validators can modify the arguments
 
        for tool_call in tool_calls:
            self.call_tool(tool_call)
        return tool_calls 
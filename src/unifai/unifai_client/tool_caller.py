from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai.types import (
    LLMProvider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    ToolCall,
    Usage,
)
from unifai.type_conversions import standardize_tools, standardize_messages, standardize_tool_choice, standardize_response_format
from unifai.wrappers._base_llm_client import LLMClient
from unifai.exceptions.tool_errors import ToolCallExecutionError, ToolCallableNotFoundError, ToolCallInvalidArgumentsError

class ToolCaller:

    def __init__(
            self,
            # tools: dict[str, Tool],
            tool_callables: dict[str, Callable[..., Any]],
            tool_execution_error_retries: int = 0,
            # tool_argument_validators: dict[str, Callable[..., Any]],
    ):
        self.tool_callables = tool_callables
        self.tool_execution_error_retries = tool_execution_error_retries
        # self.tool_callables = {}
        # for tool_name, tool in tools.items():
        #     if tool.callable:
        #         self.tool_callables[tool_name] = tool.callable
        # self.tool_callables.update(tool_callables)
        # self.tool_argument_validators = tool_argument_validators


    def set_tool_callables(self, tool_callables: dict[str, Callable[..., Any]]):
        self.tool_callables = tool_callables

    def update_tool_callables(self, tool_callables: dict[str, Callable[..., Any]]):
        self.tool_callables.update(tool_callables)

    # def set_tool_argument_validators(self, tool_argument_validators: dict[str, Callable[..., Any]]):
    #     self.tool_argument_validators = tool_argument_validators
    
    # def update_tool_argument_validators(self, tool_argument_validators: dict[str, Callable[..., Any]]):
    #     self.tool_argument_validators.update(tool_argument_validators)

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
                return tool_call
            except Exception as e:
                execution_retries += 1
                if execution_retries >= self.tool_execution_error_retries:
                    raise ToolCallExecutionError(
                        message=f"Error executing tool '{tool_name}'",
                        tool_call=tool_call,
                        original_exception=e,
                    )
        # TODO raise ToolCallExecutionError if retries are exceeded
        return tool_call


    def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        for tool_call in tool_calls:
            self.call_tool(tool_call)                   
        return tool_calls        

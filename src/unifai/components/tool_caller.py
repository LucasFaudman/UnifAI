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
from unifai.adapters._base_llm_client import LLMClient
from unifai.exceptions.tool_errors import ToolCallExecutionError, ToolCallableNotFoundError, ToolCallInvalidArgumentsError

from .concurrent_executor import ConcurrentExecutor

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

    def call_tool(self, tool_call: ToolCall) -> ToolCall:
        tool_name = tool_call.tool_name
        arguments = tool_call.arguments

        if (tool_argument_validator := self.tool_argument_validators.get(tool_name)) is not None:
            try:
                arguments = tool_argument_validator(arguments)
            except Exception as e:
                raise ToolCallInvalidArgumentsError(
                    message=f"Invalid arguments for tool '{tool_name}'",
                    tool_call=tool_call,
                    original_exception=e,
                )
        
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
        # TODO raise ToolCallExecutionError if retries are exceeded
        return tool_call


    def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        for tool_call in tool_calls:
            self.call_tool(tool_call)                   
        return tool_calls        


class ConcurrentToolCaller(ToolCaller):
    def __init__(
            self,
            tool_callables: dict[str, Callable[..., Any]],
            tool_argument_validators: Optional[dict[str, Callable[..., Any]]] = None,
            tools: Optional[list[Tool]] = None,
            tool_execution_error_retries: int = 0,            
            concurrency_type: Optional[Literal["thread", "process", "main", False]] = "thread",
            max_workers: Optional[int] = None,
            chunksize: int = 1,
            timeout: Optional[int] = None,
            shutdown: bool = True,
            wait: bool = True,
            cancel_pending: bool = False,
            executor: Optional[Any] = None,
            executor_init_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            tool_callables=tool_callables,
            tool_argument_validators=tool_argument_validators,
            tools=tools,
            tool_execution_error_retries=tool_execution_error_retries,
        )
        self.executor = ConcurrentExecutor(
            concurrency_type=concurrency_type,
            max_workers=max_workers,
            chunksize=chunksize,
            timeout=timeout,
            shutdown=shutdown,
            wait=wait,
            cancel_pending=cancel_pending,
            executor=executor,
            **(executor_init_kwargs or {}),
        )


    def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        return list(self.executor.map(self.call_tool, tool_calls))

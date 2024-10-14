from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from ...types import Tool, ToolCall
from ...exceptions.tool_errors import ToolCallExecutionError, ToolCallableNotFoundError, ToolCallInvalidArgumentsError
from ..concurrent_executor import ConcurrentExecutor
from .tool_caller import ToolCaller

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

from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from ...types import Tool, ToolCall
from ...exceptions.tool_errors import ToolCallExecutionError, ToolCallableNotFoundError, ToolCallArgumentValidationError
from ..executors.concurrent_executor import ConcurrentExecutor
from .._base_components._base_tool_caller import ToolCaller

class ConcurrentToolCaller(ToolCaller):
    def __init__(
            self,
            tool_callables: dict[str, Callable[..., Any]],
            tool_argument_validators: Optional[dict[str, Callable[..., Any]]] = None,
            tools: Optional[list[Tool]] = None,
            tool_execution_error_retries: int = 0,            
            concurrency_type: Optional[Literal["thread", "process", "main", False]] = "thread",
            max_workers: Optional[int] = None, # Default is None, which means use the number of CPUs
            chunk_size: int = 1,
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
            results_order="submitted", # Preserve order of results as submitted tool calls
            max_workers=max_workers,
            chunk_size=chunk_size,
            timeout=timeout,
            shutdown=shutdown,
            wait=wait,
            cancel_pending=cancel_pending,
            executor=executor,
            **(executor_init_kwargs or {}),
        )


    def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        # Validate all arguments before calling any tools
        for i, validated_args in enumerate(self.executor.map(self.validate_arguments, tool_calls)):
            tool_calls[i].arguments = validated_args # Validators can modify the arguments
        return list(self.executor.map(self.call_tool, tool_calls))

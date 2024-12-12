from typing import TYPE_CHECKING, Any, Literal, Optional, Type, Generic, TypeVar, overload
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..configs import UnifAIConfig
    from pathlib import Path

from ..type_conversions import standardize_config
from ..configs.function_config import FunctionConfig, InputT, OutputT, ReturnT
from ..components.functions import Function

from ._base_client import BaseClient
from ._chat_client import UnifAIChatClient
from ._rag_client import UnifAIRAGClient


class UnifAIFunctionClient(UnifAIChatClient, UnifAIRAGClient):

    def function(self, config: FunctionConfig[InputT, OutputT, ReturnT], **init_kwargs) -> Function[InputT, OutputT, ReturnT]:
        return Function(config=config, tool_registry=self._tools, _get_component=self._get_component, **init_kwargs)

    def configure(
        self,
        config: Optional["UnifAIConfig|dict[str, Any]|str|Path"] = None,
        api_keys: Optional["dict[ProviderName, str]"] = None,
        **kwargs
    ) -> None:
        BaseClient.configure(self, config, api_keys, **kwargs)
        self._init_tools()
        self._init_rag_configs()
        self._init_function_configs()
       
    def _init_function_configs(self) -> None:
        self._function_configs: dict[str, FunctionConfig] = {}
        if self.config.function_configs:
            self.register_function_configs(*self.config.function_configs)       

    def register_function_configs(self, *function_configs: FunctionConfig|dict) -> None:
        for _function_config in function_configs:
            _function_config = standardize_config(_function_config, FunctionConfig)
            self._function_configs[_function_config.name] = _function_config

    def get_function_config(self, name: "ComponentName") -> FunctionConfig:
        if (function_config := self._function_configs.get(name)) is None:
            raise KeyError(f"Function config '{name}' not found in self.function_configs")
        return function_config

    def _cleanup_function_configs(self) -> None:
        self._function_configs.clear()

    def cleanup(self) -> None:
        BaseClient.cleanup(self)
        self._cleanup_tools()
        self._cleanup_rag_configs()
        self._cleanup_function_configs()


   

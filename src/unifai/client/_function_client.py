from typing import TYPE_CHECKING, Any, Literal, Optional, Type, Generic, TypeVar, overload
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_tool_caller import ToolCaller

from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, BaseModel
from ..types import (

    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    Embeddings,
    Embedding,
    QueryResult,    
    
    ReturnOnInput,
    ResponseFormatInput,
    ToolChoiceInput,
)
from ..type_conversions import standardize_configs, standardize_config


from ._base_client import BaseClient, Path
from ._chat_client import UnifAIChatClient
from ._rag_client import UnifAIRAGClient

from ..configs import UnifAIConfig
from ..configs.function_config import FunctionConfig, InputT, OutputT, ReturnT
from ..components.function import Function



class UnifAIFunctionClient(UnifAIChatClient, UnifAIRAGClient):

    def function(self, config: FunctionConfig[InputT, OutputT, ReturnT], **init_kwargs) -> Function[InputT, OutputT, ReturnT]:
        return Function(config=config, tool_registry=self._tools, _get_component=self._get_component, **init_kwargs)

    def configure(
        self,
        config: Optional[UnifAIConfig|dict[str, Any]|str|Path] = None,
        api_keys: Optional[dict[ProviderName, str]] = None,
        **kwargs
    ) -> None:
        BaseClient.configure(self, config, api_keys, **kwargs)
        self._init_tools(self.config.tools, self.config.tool_callables)
        # self._init_function_configs(self.config.function_configs)
       
    def _init_function_configs(self, function_configs: Optional[dict[str, FunctionConfig|dict[str, Any]]] = None) -> None:
        self._function_configs: dict[str, FunctionConfig] = {}
        if function_configs:
            self.register_function_configs(function_configs)       

    # def _init_function_configs(self, function_configs: Optional[dict[str, FunctionConfig|dict[str, Any]]] = None) -> None:
    #     self._function_configs: dict[str, FunctionConfig] = {}
    #     if function_configs:
    #         self.register_function_configs(function_configs)

    # def _cleanup_function_configs(self) -> None:
    #     self._function_configs.clear()

    # def cleanup(self) -> None:
    #     BaseClient.cleanup(self)
    #     self._cleanup_tools()
    #     self._cleanup_rag_configs()
    #     self._cleanup_function_configs()

    def register_function_config(self, name: str, function_config: FunctionConfig|dict) -> None:
        function_config = standardize_config(function_config, FunctionConfig)
        self._function_configs[name] = function_config

    def register_function_configs(self, function_configs: dict[str, FunctionConfig|dict[str, Any]]) -> None:
        for name, function_config in function_configs.items():
            self.register_function_config(name, function_config)

    def get_function_config(self, name: str) -> FunctionConfig:
        if (function_config := self._function_configs.get(name)) is None:
            raise KeyError(f"Function config '{name}' not found in self.function_configs")
        return function_config

   

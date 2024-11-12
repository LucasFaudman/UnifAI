from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.tool_callers._base_tool_caller import ToolCaller


from ..types import (
    LLMProvider,
    EmbeddingProvider,
    VectorDBProvider,
    RerankProvider,
    Provider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    Embeddings,
    Embedding,
    VectorDBQueryResult,    
    
    ReturnOnInput,
    ResponseFormatInput,
    ToolChoiceInput,
)
from ..type_conversions import standardize_configs, standardize_config


from ._base_client import BaseClient, Config, ProviderConfig, Path
from ._chat_client import UnifAIChatClient
from ._rag_client import UnifAIRAGClient

from .rag_engine import RAGEngine, RAGConfig
from .function import UnifAIFunction, FunctionConfig



class UnifAIFunctionClient(UnifAIChatClient, UnifAIRAGClient):
    def __init__(
        self,
        config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
        provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
        api_keys: Optional[dict[str, str]] = None,
        default_providers: Optional[dict[str, str]] = None,  
        tools: Optional[Sequence[ToolInput]] = None,
        tool_callables: Optional[dict[str, Callable]] = None,
        rag_configs: Optional[dict[str, RAGConfig|dict[str, Any]]] = None,
        function_configs: Optional[dict[str, FunctionConfig|dict[str, Any]]] = None,              
        **kwargs
    ):
        BaseClient.__init__(
            self,
            config_obj_dict_or_path=config_obj_dict_or_path,
            provider_configs=provider_configs,
            api_keys=api_keys,
            default_providers=default_providers,
            **kwargs
        )
        self._init_tools(tools, tool_callables)
        self._init_rag_configs(rag_configs)        

       

    def _init_function_configs(self, function_configs: Optional[dict[str, FunctionConfig|dict[str, Any]]] = None) -> None:
        self._function_configs: dict[str, FunctionConfig] = {}
        if function_configs:
            self.register_function_configs(function_configs)

    def _cleanup_function_configs(self) -> None:
        self._function_configs.clear()

    def cleanup(self) -> None:
        BaseClient.cleanup(self)
        self._cleanup_tools()
        self._cleanup_rag_configs()
        self._cleanup_function_configs()

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

   
    def get_function(
            self, 
            config_obj_or_name: Optional[FunctionConfig | str] = None,
            tool_caller_instance: Optional["ToolCaller"] = None,
            rag_engine_instance: Optional[RAGEngine] = None,
            **kwargs
            ) -> UnifAIFunction:
        
        
        if isinstance(config_obj_or_name, str):
            function_config = self.get_function_config(config_obj_or_name)
        elif isinstance(config_obj_or_name, FunctionConfig):
            function_config = config_obj_or_name.model_copy(update=kwargs, deep=True) if kwargs else config_obj_or_name
        elif config_obj_or_name is None:
            function_config = FunctionConfig(**kwargs)
        else:
            raise ValueError(f"Invalid config_obj_or_name: {config_obj_or_name}. Must be a FunctionConfig object or a string (name of a registered FunctionConfig)")
        
        if tool_caller_instance is not None:
            function_config.tool_caller_class_or_instance = tool_caller_instance
        if rag_engine_instance is None and function_config.rag_config:
            rag_engine_instance = self.get_rag_engine(function_config.rag_config)
        
        return UnifAIFunction(
            config=function_config,
            rag_engine=rag_engine_instance,
            get_llm_client=self.get_llm_client,
            tool_registry=self._tools,
        )
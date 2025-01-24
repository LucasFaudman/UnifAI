from typing import TYPE_CHECKING, Any, Literal, Optional, Type, Generic, TypeVar, overload, ParamSpec, Unpack
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..configs import UnifAIConfig
    from pathlib import Path

from ..utils import combine_dicts
from ..type_conversions import standardize_config
from ..configs.function_config import FunctionConfig, InputP, InputReturnT, OutputT, ReturnT
from ..configs.input_parser_config import InputParserConfig
from ..configs.output_parser_config import OutputParserConfig

from ..components.input_parsers import InputParser
from ..components.output_parsers import OutputParser
from ..components.functions import Function

from ._base_client import BaseClient
from ._chat_client import UnifAIChatClient
from ._rag_client import UnifAIRAGClient


class UnifAIFunctionClient(UnifAIChatClient, UnifAIRAGClient):

    def input_parser(
            self,
            provider_config_or_name: "ProviderName | InputParserConfig[InputP,InputReturnT] | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "InputParser[InputP,InputReturnT]":
        return self._get_component("input_parser", provider_config_or_name, init_kwargs)

    def output_parser(
            self, 
            provider_config_or_name: "ProviderName | OutputParserConfig[OutputT,ReturnT] | tuple[ProviderName, ComponentName]" = "default",          
            **init_kwargs
            ) -> "OutputParser[OutputT,ReturnT]":
        return self._get_component("output_parser", provider_config_or_name, init_kwargs)
        
    # def function(
    #         self, 
    #         config: "FunctionConfig[InputP, InputReturnT, OutputT, ReturnT]", 
    #         **init_kwargs
    #     ) -> "Function[InputP, InputReturnT, OutputT, ReturnT]":
    #     default_init_kwargs = {
    #         "tool_registry": self._tools,
    #         "_get_component": self._get_component,
    #         "_get_input_parser": self.input_parser,
    #         "_get_output_parser": self.output_parser,
    #         "_get_ragpipe": self.ragpipe,
    #         "_get_function": self.function
    #     }
    #     _init_kwargs = combine_dicts(default_init_kwargs, init_kwargs)
    #     return Function(config=config, **_init_kwargs)

    def function(
            self, 
            provider_config_or_name: "ProviderName | FunctionConfig[InputP, InputReturnT, OutputT, ReturnT] | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
        ) -> "Function[InputP, InputReturnT, OutputT, ReturnT]":
        default_init_kwargs = {
            "tool_registry": self._tools,
            "_get_component": self._get_component,
            "_get_input_parser": self.input_parser,
            "_get_output_parser": self.output_parser,
            "_get_ragpipe": self.ragpipe,
            "_get_function": self.function
        }
        _init_kwargs = combine_dicts(default_init_kwargs, init_kwargs)
        return self._get_component("function", provider_config_or_name, _init_kwargs)    

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


   

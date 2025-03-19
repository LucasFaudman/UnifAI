from typing import TYPE_CHECKING, get_type_hints, cast, Any, Literal, Optional, Type, Generic, TypeVar, overload, ParamSpec, Unpack
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    # from ..types.annotations import * #ComponentName, ProviderName, InputP, InputReturnT, OutputT, ReturnT
    from ..configs import UnifAIConfig
    from pathlib import Path

    from ..configs.input_parser_config import InputParserConfig
    from ..configs.output_parser_config import OutputParserConfig
    from ..configs.rag_config import RAGConfig
    from ..configs.tool_caller_config import ToolCallerConfig
    from ..configs.llm_config import LLMConfig    

    from ..components._base_components._base_input_parser import InputParser
    from ..components._base_components._base_output_parser import OutputParser
    from ..components._base_components._base_llm import LLM
    from ..components._base_components._base_tool_caller import ToolCaller
    from ..components.functions import Function
    from ..components.prompt_templates import PromptModel

    from ..types import Message, Tool


from ..exceptions import UnifAIError, ToolChoiceError
from ..types.annotations import * #ComponentName, ProviderName, InputP, InputReturnT, OutputT, ReturnT

from ..utils import combine_dicts, clean_locals, copy_paramspec_from, copy_init_from, update_kwargs_with_locals
from ..type_conversions import standardize_config
from ..configs.function_config import FunctionConfig, convert_input_to_user_message, return_last_message

from ._base_client import BaseClient
from ._chat_client import UnifAIChatClient
from ._rag_client import UnifAIRAGClient


class UnifAIFunctionClient(UnifAIChatClient, UnifAIRAGClient):

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
        self._function_configs: "dict[str, FunctionConfig]" = {}
        if self.config.function_configs:
            self.register_function_configs(*self.config.function_configs)       

    def register_function_configs(self, *function_configs: "FunctionConfig | dict") -> None:
        for _function_config in function_configs:
            _function_config = standardize_config(_function_config, FunctionConfig)
            self._function_configs[_function_config.name] = _function_config

    def get_function_config(self, name: "ComponentName") -> "FunctionConfig":
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



    # Input Parsers
    def _get_input_parser(
            self,
            config_or_name: "InputParserConfig[InputP, InputReturnT] | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "InputParser[InputP,InputReturnT]":
        return self._get_component("input_parser", config_or_name, init_kwargs, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists)
    
    def input_parser_from_config(
            self,
            config: "InputParserConfig[InputP, InputReturnT]",
            *,
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "InputParser[InputP, InputReturnT]":
        return self._get_input_parser(config, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)

    def input_parser_from_name(
            self,
            *,
            name: "ComponentName" = "default",
            provider: "ProviderName" = "default",
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "InputParser":
        return self._get_input_parser((provider, name), cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)

    def input_parser(
            self,
            *,
            name: "ComponentName" = "__name__",
            provider: "ProviderName" = "default",
            callable: Optional[Callable[InputP, InputReturnT | Callable[..., InputReturnT]]] = None,
            return_type: Optional[Type[InputReturnT]] = None,            
            extra_kwargs: Optional[dict[Literal["parse_input"], dict[str, Any]]] = None,
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "InputParser[InputP, InputReturnT]":
        config = self._config_from_locals('input_parser', provider, locals())
        return self._get_input_parser(config, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)
 
    # Output Parsers
    def _get_output_parser(
            self,
            config_or_name: "OutputParserConfig[OutputT, ReturnT] | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "OutputParser[OutputT, ReturnT]":
        return self._get_component("output_parser", config_or_name, init_kwargs, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists)
    
    def output_parser_from_config(
            self,
            config: "OutputParserConfig[OutputT, ReturnT]",
            *,
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "OutputParser[OutputT, ReturnT]":
        return self._get_output_parser(config, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)    

    def output_parser_from_name(
            self,
            *,
            name: "ComponentName" = "default",
            provider: "ProviderName" = "default",
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "OutputParser":
        return self._get_output_parser((provider, name), cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)        

    def output_parser(
            self,
            *,
            name: "ComponentName" = "__name__",
            provider: "ProviderName" = "default",
            callable: Optional[Callable[[OutputT], ReturnT]] = None,
            output_type: Optional[Type[OutputT]] = None,
            return_type: Optional[Type[ReturnT]] = None,
            extra_kwargs: Optional[dict[Literal["parse_output"], dict[str, Any]]] = None,
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "OutputParser[OutputT, ReturnT]":
        config = self._config_from_locals('output_parser', provider, locals())
        return self._get_output_parser(config, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)

    # Functions
    def _get_function(
            self,
            config_or_name: "FunctionConfig[InputP, InputReturnT, OutputT, ReturnT] | ProviderName | tuple[ProviderName, ComponentName]",
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "Function[InputP, InputReturnT, OutputT, ReturnT]":
        if "tool_registry" not in init_kwargs:
            init_kwargs["tool_registry"] = self._tools
        return self._get_component("function", config_or_name, init_kwargs, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists)

    def function_from_config(
            self,
            config: "FunctionConfig[InputP, InputReturnT, OutputT, ReturnT]",
            *,
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "Function[InputP, InputReturnT, OutputT, ReturnT]":
        return self._get_function(config, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)

    def function_from_name(
            self,
            *,
            name: "ComponentName" = "default",
            provider: "ProviderName" = "default",
            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "Function":
        return self._get_function((provider, name), cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)

    def function(
            self,
            *,
            name: "ComponentName",
            provider: "ProviderName" = "default",
            stateless: bool = True,
            input_parser: "Callable[InputP, InputReturnT] | Callable[InputP, Callable[..., InputReturnT]] | InputParserConfig[InputP, InputReturnT] | RAGConfig[..., InputP] | FunctionConfig[InputP, Any, Any, InputReturnT]" = convert_input_to_user_message,
            output_parser: "Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] | Type[ReturnT] | FunctionConfig[..., Any, Any, ReturnT]" = return_last_message,
            structured_outputs_mode: Literal["tool_call", "json_schema"] = "tool_call",
            llm: "LLM | LLMConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            llm_model: "Optional[ModelName]" = None,

            system_prompt: "Optional[str | Callable[..., str] | PromptModel | Type[PromptModel]]" = None,
            system_prompt_kwargs: "Optional[dict[str, Any]]" = None,
            examples: "Optional[list[Message | dict[Literal['input', 'response'], Any]]]" = None,
    
            tools: "Optional[list[ToolInput]]" = None,
            tool_choice: "Optional[ToolName | Tool | Literal['auto', 'required', 'none'] | list[ToolName | Tool | Literal['auto', 'required', 'none']]]" = None,
            enforce_tool_choice: bool = True,
            tool_callables: "Optional[dict[ToolName, Callable[..., Any]]]" = None,
            tool_caller: "Optional[ToolCaller | ToolCallerConfig | ProviderName | tuple[ProviderName, ComponentName]]" = "default",

            response_format: "Optional[Literal['text', 'json'] | dict[Literal['json_schema'], dict[str, str] | Type[BaseModel] | Tool]]" = None,
            return_on: "Literal['content', 'tool_call', 'message'] | ToolName | Tool | list[ToolName | Tool]" = "content",

            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,

            max_messages_per_run: int = 10,
            max_tool_calls_per_run: Optional[int] = None,
            max_tokens_per_run: Optional[int] = None,
            max_input_tokens_per_run: Optional[int] = None,
            max_output_tokens_per_run: Optional[int] = None,
            count_tokens_proactively: bool = False,

            error_retries: dict[Type[UnifAIError | Exception], int] = { ToolChoiceError: 3 },
            error_handlers: dict[Type[UnifAIError | Exception], Callable] = {},
            extra_kwargs: Optional[dict[Literal["chat", "system_prompt", "run", "run_stream"], dict[str, Any]]] = None,

            cache: bool = True,
            create_if_not_exists: bool = True,
            reuse_if_exists: bool = True,
            override_config_if_exists: bool = True,
            **init_kwargs
            ) -> "Function[InputP, InputReturnT, OutputT, ReturnT]":
        config = self._config_from_locals('function', provider, locals())
        return self._get_function(config, cache, create_if_not_exists, reuse_if_exists, override_config_if_exists, **init_kwargs)
        

    # @copy_init_from(FunctionConfig.__init__)
    # def function(
    #     self, 
    #     input_parser: "Callable[InputP, InputReturnT] | Callable[InputP, Callable[..., InputReturnT]] | InputParserConfig[InputP, InputReturnT] | RAGConfig[..., InputP] | FunctionConfig[InputP, Any, Any, InputReturnT]  | FunctionConfig" = convert_input_to_user_message,
    #     output_parser: "Type[ReturnT] | Callable[[OutputT], ReturnT] | OutputParserConfig[OutputT, ReturnT] | FunctionConfig[..., Any, Any, ReturnT] | FunctionConfig" = return_last_message,
    #     **kwargs
    # ) -> "Function[InputP, InputReturnT, OutputT, ReturnT]":
    #     config = FunctionConfig(input_parser=input_parser, output_parser=output_parser, **kwargs)
    #     function = self.function_from_config(config, **kwargs)
    #     return function

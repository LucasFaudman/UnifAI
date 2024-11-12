from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.llms._base_llm_client import LLMClient


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
from ..type_conversions import standardize_tool, standardize_tools

from ..components.tool_callers._base_tool_caller import ToolCaller
from .chat import Chat
from ._base_client import BaseClient, Config, ProviderConfig, Path


class UnifAIChatClient(BaseClient):

    def get_llm_client(self, provider: Optional[str] = None, **client_kwargs) -> "LLMClient":
        return self._get_component(provider, "llm", **client_kwargs)

    def __init__(
        self,
        config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
        provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
        api_keys: Optional[dict[str, str]] = None,
        default_providers: Optional[dict[str, str]] = None,
        tools: Optional[Sequence[ToolInput]] = None,
        tool_callables: Optional[dict[str, Callable]] = None,        
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


    def _init_tools(self, tools: Optional[Sequence[ToolInput]] = None, tool_callables: Optional[dict[str, Callable]] = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._tool_callables: dict[str, Callable] = {} 
        # Maybe TODO Tools in Config?
        if tools:
            self.register_tools(tools, tool_callables)

    def _cleanup_tools(self) -> None:
        self._tools.clear()
        self._tool_callables.clear()

    def cleanup(self) -> None:
        BaseClient.cleanup(self)
        self._cleanup_tools()

    # Tools
    def register_tool(self, tool: ToolInput, tool_callable: Optional[Callable] = None, name: Optional[str] = None) -> None:
        tool = standardize_tool(tool)
        tool_name = name or tool.name
        self._tools[tool_name] = tool
        if tool_callable:
            self.register_tool_callable(tool_name, tool_callable)

    def register_tool_callable(self, tool_name: str, tool_callable: Callable) -> None:
        self._tool_callables[tool_name] = tool_callable

    def register_tools(self, tools: Sequence[ToolInput], tool_callables: Optional[dict[str, Callable]] = None) -> None:
        for tool_name, tool in standardize_tools(tools).items():
            self.register_tool(tool, tool_callables.get(tool_name) if tool_callables else None)

    def register_tool_callables(self, tool_callables: dict[str, Callable]) -> None:
        self._tool_callables.update(tool_callables)

    def get_tool(self, tool_name: str) -> Tool:
        return self._tools[tool_name]
        
    def get_tool_callable(self, tool_name: str) -> Callable:
        return self._tool_callables[tool_name]
    
    # Chat
    def start_chat(
        self,
        messages: Optional[Sequence[MessageInput]] = None,
        provider: Optional[LLMProvider] = None,            
        model: Optional[str] = None,

        system_prompt: Optional[str] = None,
        return_on: ReturnOnInput = "content",
        response_format: Optional[ResponseFormatInput] = None,
        
        tools: Optional[Sequence[ToolInput]] = None,            
        tool_choice: Optional[ToolChoiceInput] = None,
        enforce_tool_choice: bool = True,
        tool_choice_error_retries: int = 3,
        tool_callables: Optional[dict[str, Callable]] = None,
        tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller,
        tool_caller_kwargs: Optional[dict[str, Any]] = None,
        tool_registry: Optional[dict[str, Tool]] = None,

        max_messages_per_run: int = 10,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None, 
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Chat:
                
        return Chat(
            get_llm_client=self.get_llm_client,

            provider=provider,
            model=model,
            
            messages=messages,
            system_prompt=system_prompt,
            return_on=return_on,
            response_format=response_format,
            
            tools=tools,
            tool_choice=tool_choice,
            enforce_tool_choice=enforce_tool_choice,
            tool_choice_error_retries=tool_choice_error_retries,
            tool_callables=tool_callables,
            tool_caller_class_or_instance=tool_caller_class_or_instance,
            tool_caller_kwargs=tool_caller_kwargs,
            tool_registry=tool_registry or self._tools,

            max_messages_per_run=max_messages_per_run,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


    def chat(
        self,
        messages: Optional[Sequence[MessageInput]] = None,
        provider: Optional[LLMProvider] = None,            
        model: Optional[str] = None,

        system_prompt: Optional[str] = None,
        return_on: ReturnOnInput = "content",
        response_format: Optional[ResponseFormatInput] = None,
        
        tools: Optional[Sequence[ToolInput]] = None,            
        tool_choice: Optional[ToolChoiceInput] = None,
        enforce_tool_choice: bool = True,
        tool_choice_error_retries: int = 3,
        tool_callables: Optional[dict[str, Callable]] = None,
        tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller,
        tool_caller_kwargs: Optional[dict[str, Any]] = None,
        tool_registry: Optional[dict[str, Tool]] = None,

        max_messages_per_run: int = 10,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None, 
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Chat:
        chat = self.start_chat(
            messages=messages,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            return_on=return_on,
            response_format=response_format,            
            tools=tools,
            tool_choice=tool_choice,
            enforce_tool_choice=enforce_tool_choice,
            tool_choice_error_retries=tool_choice_error_retries,
            tool_callables=tool_callables,
            tool_caller_class_or_instance=tool_caller_class_or_instance,
            tool_caller_kwargs=tool_caller_kwargs,
            tool_registry=tool_registry,
            max_messages_per_run=max_messages_per_run,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if messages:
            chat.run(**kwargs)
        return chat
        

    def chat_stream(
        self,
        messages: Optional[Sequence[MessageInput]] = None,
        provider: Optional[LLMProvider] = None,            
        model: Optional[str] = None,

        system_prompt: Optional[str] = None,
        return_on: ReturnOnInput = "content",
        response_format: Optional[ResponseFormatInput] = None,
        
        tools: Optional[Sequence[ToolInput]] = None,            
        tool_choice: Optional[ToolChoiceInput] = None,
        enforce_tool_choice: bool = True,
        tool_choice_error_retries: int = 3,
        tool_callables: Optional[dict[str, Callable]] = None,
        tool_caller_class_or_instance: Optional[Type[ToolCaller]|ToolCaller] = ToolCaller,
        tool_caller_kwargs: Optional[dict[str, Any]] = None,
        tool_registry: Optional[dict[str, Tool]] = None,

        max_messages_per_run: int = 10,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None, 
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> Generator[MessageChunk, None, Chat]:
        chat = self.start_chat(
            messages=messages,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            return_on=return_on,
            response_format=response_format,            
            tools=tools,
            tool_choice=tool_choice,
            enforce_tool_choice=enforce_tool_choice,
            tool_choice_error_retries=tool_choice_error_retries,
            tool_callables=tool_callables,
            tool_caller_class_or_instance=tool_caller_class_or_instance,
            tool_caller_kwargs=tool_caller_kwargs,
            tool_registry=tool_registry,
            max_messages_per_run=max_messages_per_run,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        if messages:
            yield from chat.run_stream(**kwargs)
        return chat

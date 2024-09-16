from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union

from unifai.ai_client_wrappers import BaseAIClientWrapper
from unifai.types import (
    AIProvider, 
    EvaluateParameters,
    EvaluateParametersInput, 
    Message,
    MessageInput, 
    Tool,
    ToolInput
)
from unifai.type_conversions import make_few_shot_prompt, standardize_eval_prameters, standardize_tools
from .chat import Chat

class UnifAIClient:
    TOOLS: list[ToolInput] = []
    TOOL_CALLABLES: dict[str, Callable] = {}
    EVAL_PARAMETERS: list[EvaluateParametersInput] = []


    def import_client_wrapper(self, provider: AIProvider) -> Type[BaseAIClientWrapper]:
        match provider:
            case "anthropic":
                from unifai.ai_client_wrappers import AnthropicWrapper
                return AnthropicWrapper
            case "openai":
                from unifai.ai_client_wrappers import OpenAIWrapper
                return OpenAIWrapper
            case "ollama":
                from unifai.ai_client_wrappers import OllamaWrapper
                return OllamaWrapper

    
    def __init__(self, 
                 provider_client_kwargs: Optional[dict[AIProvider, dict[str, Any]]] = None,
                 tools: Optional[list[ToolInput]] = None,
                 tool_callables: Optional[dict[str, Callable]] = None,
                 eval_prameters: Optional[list[EvaluateParametersInput]] = None

                 ):
        self.provider_client_kwargs = provider_client_kwargs if provider_client_kwargs is not None else {}
        self.providers = list(self.provider_client_kwargs.keys())
        self.default_provider: AIProvider = self.providers[0] if len(self.providers) > 0 else "openai"
        
        self._clients: dict[AIProvider, BaseAIClientWrapper] = {}
        self.tools: dict[str, Tool] = {}
        self.tool_callables: dict[str, Callable] = {}
        self.eval_prameters: dict[str, EvaluateParameters] = {}
        
        self.add_tools(tools or self.TOOLS)
        self.add_tool_callables(tool_callables)
        self.add_eval_prameters(eval_prameters or self.EVAL_PARAMETERS)
    

    def add_tools(self, tools: Optional[list[ToolInput]]):
        if not tools: return

        for tool_name, tool in standardize_tools(tools).items():
            self.tools[tool_name] = tool
            if (tool_callable := getattr(tool, "callable", None)) is not None:
                self.tool_callables[tool_name] = tool_callable

    def add_tool_callables(self, tool_callables: Optional[dict[str, Callable]]):
        if not tool_callables: return
        self.tool_callables.update(tool_callables)


    def add_eval_prameters(self, eval_prameters: Optional[list[EvaluateParametersInput]]):
        if not eval_prameters: return
        self.eval_prameters.update(standardize_eval_prameters(eval_prameters))

        
    def init_client(self, provider: AIProvider, **client_kwargs) -> BaseAIClientWrapper:
        client_kwargs = {**self.provider_client_kwargs[provider], **client_kwargs}
        self._clients[provider] = self.import_client_wrapper(provider)(**client_kwargs)
        return self._clients[provider]
    

    def get_client(self, provider: Optional[AIProvider] = None) -> BaseAIClientWrapper:
        provider = provider or self.default_provider
        if provider not in self._clients:
            return self.init_client(provider)
        return self._clients[provider]


    # List Models
    def list_models(self, provider: Optional[AIProvider] = None) -> list[str]:
        return self.get_client(provider).list_models()


    # def filter_tools_by_tool_choice(self, tools: list[Tool], tool_choice: str) -> list[Tool]:
    #     if tool_choice == "auto" or tool_choice == "required":
    #         return tools
    #     if tool_choice == "none":
    #         return []
    #     return [tool for tool in tools if tool.name == tool_choice]

    
    def chat(
            self,
            messages: Optional[Sequence[MessageInput]] = None,
            provider: Optional[AIProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,             
            tools: Optional[Sequence[ToolInput]] = None,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,

            return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            enforce_tool_choice: bool = False,
            tool_choice_error_retries: int = 3,

            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            stream: bool = False,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,

            **kwargs
            ):
            chat = Chat(
                parent=self,
                provider=provider or self.default_provider,
                messages=messages if messages is not None else [],
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                tool_callables=tool_callables,
                tool_choice=tool_choice,
                response_format=response_format,
                return_on=return_on,
                enforce_tool_choice=enforce_tool_choice,
                tool_choice_error_retries=tool_choice_error_retries,

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
        

    def evaluate(self, 
                 eval_type: str | EvaluateParameters, 
                 content: Any, 
                 provider: Optional[AIProvider] = None,
                 model: Optional[str] = None,
                 **kwargs
                 ) -> Any:
        
        # Get eval_parameters from eval_type
        if isinstance(eval_type, str):
            if (eval_parameters := self.eval_prameters.get(eval_type)) is None:
                raise ValueError(f"Eval type '{eval_type}' not found in eval_prameters")
        elif isinstance(eval_type, EvaluateParameters):
            eval_parameters = eval_type
        else:
            raise ValueError(
                f"Invalid eval_type: {eval_type}. Must be a string (eval_type of EvaluateParameters in self.EVAL_PARAMETERS) or an EvaluateParameters object")

        # Determine return_on parameter from eval_parameters and tool_choice
        if eval_parameters.return_on:
            # Use the return_on parameter from eval_parameters if provided
            return_on = eval_parameters.return_on
        elif isinstance(eval_parameters.tool_choice, str):
            # Use the tool_choice parameter from eval_parameters if its a string (single tool name)
            return_on = eval_parameters.tool_choice
        elif eval_parameters.tool_choice:
            # Use the last tool choice if tool_choice is a non-empty sequence of tool names (tool_choice queue)
            return_on = eval_parameters.tool_choice[-1]
        else:
            # Default to return on content if no return_on or tool_choice is provided
            return_on = "content"

        # Create input messages from system_prompt, few-shot examples, and content
        input_messages = make_few_shot_prompt(
            system_prompt=eval_parameters.system_prompt,
            examples=eval_parameters.examples,
            content=content
        )

        # Initialize and run chat
        chat = self.chat(
            messages=input_messages,
            provider=provider,
            model=model,
            tools=eval_parameters.tools,
            tool_choice=eval_parameters.tool_choice,
            response_format=eval_parameters.response_format,
            return_on=return_on,
            **kwargs
        )
        
        # Return the desired attribute of the chat object or the chat object itself based on eval_parameters.return_as
        return getattr(chat, eval_parameters.return_as) if eval_parameters.return_as != "chat" else chat
    
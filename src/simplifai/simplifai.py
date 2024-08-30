

from typing import Optional, Union, Any, Literal, Mapping, Type, Callable, Collection, Sequence, Type, List, Dict, Tuple

from json import dumps as json_dumps
from .baseaiclientwrapper import BaseAIClientWrapper

from ._types import Message, Tool, ToolCall, tool_from_dict

AIProvider = Literal["anthropic", "openai", "ollama"]

class SimplifAI:
    TOOL_CALLABLES = {}


    def import_client_wrapper(self, provider: AIProvider) -> Type[BaseAIClientWrapper]:
        match provider:
            case "anthropic":
                from .anthropic_wrapper import AnthropicWrapper
                return AnthropicWrapper
            case "openai":
                from .openai_wrapper import OpenAIWrapper
                return OpenAIWrapper
            case "ollama":
                from .ollama_wrapper import OllamaWrapper
                return OllamaWrapper

    
    def __init__(self, 
                 provider_client_kwargs: Optional[dict[AIProvider, dict[str, Any]]] = None,
                 tool_callables: Optional[dict[str, Callable]] = None                 
                 ):
        self.provider_client_kwargs = provider_client_kwargs if provider_client_kwargs is not None else {}
        self.providers = list(self.provider_client_kwargs.keys())
        self.default_provider: AIProvider = self.providers[0] if len(self.providers) > 0 else "openai"
        self._clients: dict[AIProvider, BaseAIClientWrapper] = {}
        self.tool_callables = tool_callables or self.TOOL_CALLABLES
        
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


    def do_tool_call(self, tool_call: ToolCall, client: BaseAIClientWrapper) -> tuple[Message, Any]:
        tool_name = tool_call.tool_name
        tool_callable = self.tool_callables.get(tool_name)
        if tool_callable is not None:
            arguments = tool_call.arguments or {}
            tool_output = tool_callable(**arguments)
        else:
            tool_output = None
        
        std_tool_output_message = Message(
            role="tool", 
            content=client.format_content(tool_output), 
            images=None,
            tool_calls=[tool_call],
            response_object=tool_output
        )
        client_tool_output_message = client.prep_input_message(std_tool_output_message)
        return std_tool_output_message, client_tool_output_message


    # def standardize_messages(self, messages: list[Union[Message, str, dict[str, Any]]], client: BaseAIClientWrapper) -> list[Message]:
    #     for i, message in enumerate(messages):
    #         if isinstance(message, Message):
    #             continue
    #         elif isinstance(message, str):
    #             messages[i] = Message(role="user", content=message)
    #         elif type(message) == dict:
    #             # Not a TypeDict since used by clients
    #             messages[i] = Message(**message)
    #         else:
    #             messages[i] = client.extract_output_message(message)
        
    #     return messages
    def standardize_messages(self, messages: list[Union[Message, str, dict[str, Any]]]) -> list[Message]:
        std_messages = []
        for message in messages:
            if isinstance(message, Message):
                std_messages.append(message)
            elif isinstance(message, str):
                std_messages.append(Message(role="user", content=message))
            elif isinstance(message, dict):
                std_messages.append(Message(**message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")        
        return std_messages    
    
    def standardize_tools(self, tools: list[Union[Tool, dict[str, Any]]]) -> list[Tool]:
        std_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                std_tools.append(tool)
            elif isinstance(tool, dict):
                std_tools.append(tool_from_dict(tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        
        return std_tools
    
    def standardize_tool_choice(self, tool_choice: Union[Literal["auto", "required", "none"], Tool, str, dict]) -> str:
        if isinstance(tool_choice, Tool):
            return tool_choice.name
        if isinstance(tool_choice, dict):
            tool_type = tool_choice['type']
            return tool_choice[tool_type]['name']
        
        # tool_choice is a string tool_name or Literal value "auto", "required", or "none"
        return tool_choice
    
    def check_tool_choice_obeyed(self, tool_choice: str, tool_calls: Optional[list[ToolCall]]) -> bool:
        if tool_calls:
            tool_names = [tool_call.tool_name for tool_call in tool_calls]
            if (
                # tools were called but tool choice is none
                tool_choice == 'none'
                # the correct tool was not called and tool choice is not "required" (required=any one or more tools must be called) 
                or (tool_choice != 'required' and tool_choice not in tool_names)
                ):
                print(f"Tools called and tool_choice={tool_choice} NOT OBEYED")
                return False
        elif tool_choice == 'required':
            print(f"Tools NOT called and tool_choice={tool_choice} NOT OBEYED")
            return False 
        
        print(f"tool_choice={tool_choice} OBEYED")
        return True       

    # Chat
    def chat(
            self,
            messages: list[Union[Message, str, dict[str, Any]]],
            provider: Optional[AIProvider] = None,            
            model: Optional[str] = None,             
            tools: Optional[list[Union[Tool, dict[str, Any]]]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict]] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,

            return_on: Optional[Union[Literal["content", "message"], str, Collection[str]]] = "content",
            enforce_tool_choice: bool = False,
            tool_choice_error_retries: int = 3,
            
            **kwargs
            ) -> list[Message]:
        
        # import and init client
        client = self.get_client(provider)
        
        # standardize inputs and prep copies for client in its format
        # (Note to self: 2 copies are stored to prevent converting back and forth between formats on each iteration.
        # this choice is at the cost of memory but is done to prevent unnecessary conversions 
        # and allow for easier debugging and error handling.
        # May revert back to optimizing for memory later if needed.)
        model = model or client.default_model
        std_messages = self.standardize_messages(messages)
        client_messages = [client.prep_input_message(message) for message in std_messages]

        if tools:
            std_tools = self.standardize_tools(tools)
            client_tools = [client.prep_input_tool(tool) for tool in std_tools]
        else:
            std_tools = None
            client_tools = None

        if tool_choice:            
            std_tool_choice = self.standardize_tool_choice(tool_choice)
            client_tool_choice = client.prep_input_tool_choice(tool_choice)
        else:
            std_tool_choice = client_tool_choice = None
            client_tool_choice = None

        if response_format:
            client_response_format = client.prep_input_response_format(response_format)
        else:
            client_response_format = None
        
        # Chat and ToolCall handling loop
        while (response := client.chat(
                messages=client_messages,                 
                model=model, 
                tools=client_tools, 
                tool_choice=client_tool_choice,
                response_format=client_response_format, 
                **kwargs
            )):

            # TODO check if response is an APIError

            std_message, client_message = client.extract_std_and_client_messages(response)

            # Enforce Tool Choice: Check if tool choice is obeyed
            # auto:  
            if enforce_tool_choice and std_tool_choice != 'auto' and std_tool_choice is not None:
                if self.check_tool_choice_obeyed(std_tool_choice, std_message.tool_calls):
                    # TODO implement tool choice sequence
                    std_tool_choice = client_tool_choice = "auto" # set to auto for next iteration
                elif tool_choice_error_retries > 0:
                    tool_choice_error_retries -= 1
                    # Continue to next iteration without updating messages (retry)
                    continue
                else:
                    print("Tool choice error retries exceeded")
                    raise ValueError("Tool choice error retries exceeded")

            # Update messages with assistant message
            std_messages.append(std_message)
            client_messages.append(client_message)
            
            if return_on == "message":
                print("returning on message")
                break
            
            if tool_calls := std_message.tool_calls:
                # Return on tool_call before processing messages
                if any(
                    tool_call.tool_name == return_on # return_on is a tool name
                    or (isinstance(return_on, Collection) and tool_call.tool_name in return_on)
                    for tool_call in tool_calls):
                    print("returning on tool call", return_on)
                    break
                
                for tool_call in tool_calls:
                    std_tool_output_message, client_tool_output_message = self.do_tool_call(tool_call, client)
                    # Update messages with tool outputs
                    std_messages.append(std_tool_output_message)
                    client_messages.append(client_tool_output_message)
                
                # Process messages after submitting tool outputs
                continue

            print("Returning on content:", std_message.content)
            break

        return std_messages

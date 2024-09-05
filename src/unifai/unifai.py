

from typing import Optional, Union, Any, Literal, Mapping, Type, Callable, Collection, Sequence, Type, List, Dict, Tuple

from json import dumps as json_dumps
from .baseaiclientwrapper import BaseAIClientWrapper

from ._types import Message, Tool, ToolCall, EvalTypeParameters
from ._convert_types import tool_from_dict, stringify_content, make_few_shot_prompt

AIProvider = Literal["anthropic", "openai", "ollama"]

ToolInput = Union[Tool, dict[str, Any], str]
EvalTypeParametersInput = Union[EvalTypeParameters, dict[str, Any]]

class UnifAIClient:
    TOOLS: list[ToolInput] = []
    TOOL_CALLABLES: dict[str, Callable] = {}
    EVAL_TYPES: list[EvalTypeParametersInput] = []


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
                 tools: Optional[list[ToolInput]] = None,
                 tool_callables: Optional[dict[str, Callable]] = None,
                 eval_types: Optional[list[EvalTypeParametersInput]] = None

                 ):
        self.provider_client_kwargs = provider_client_kwargs if provider_client_kwargs is not None else {}
        self.providers = list(self.provider_client_kwargs.keys())
        self.default_provider: AIProvider = self.providers[0] if len(self.providers) > 0 else "openai"
        
        self._clients: dict[AIProvider, BaseAIClientWrapper] = {}
        self.tools: dict[str, Tool] = {}
        self.tool_callables: dict[str, Callable] = {}
        self.eval_types: dict[str, EvalTypeParameters] = {}
        
        self.add_tools(tools or self.TOOLS)
        self.add_tool_callables(tool_callables)
        self.add_eval_types(eval_types or self.EVAL_TYPES)
    

    def add_tools(self, tools: Optional[list[ToolInput]]):
        if not tools: return

        for tool in self.standardize_tools(tools):
            self.tools[tool.name] = tool
            if (tool_callable := getattr(tool, "callable", None)) is not None:
                self.tool_callables[tool.name] = tool_callable

    def add_tool_callables(self, tool_callables: Optional[dict[str, Callable]]):
        if not tool_callables: return
        self.tool_callables.update(tool_callables)


    def add_eval_types(self, eval_types: Optional[list[EvalTypeParametersInput]]):
        if not eval_types: return
        self.eval_types.update(self.standardize_eval_types(eval_types))

        
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

    def standardize_eval_types(self, eval_types: Sequence[EvalTypeParametersInput]) -> dict[str, EvalTypeParameters]:
        std_eval_types = {}
        for eval_type in eval_types:
            if isinstance(eval_type, EvalTypeParameters):
                std_eval_types[eval_type.name] = eval_type
            elif isinstance(eval_type, dict):
                std_eval_types[eval_type['eval_type']] = EvalTypeParameters(**eval_type)
            else:
                raise ValueError(f"Invalid eval_type type: {type(eval_type)}")
        return std_eval_types


    def standardize_messages(self, messages: Sequence[Union[Message, str, dict[str, Any]]]) -> list[Message]:
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
    
    
    def standardize_tools(self, tools: Sequence[ToolInput]) -> list[Tool]:
        std_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                std_tools.append(tool)
            elif isinstance(tool, dict):
                std_tools.append(tool_from_dict(tool))
            elif isinstance(tool, str):
                if std_tool := self.tools.get(tool):
                    std_tools.append(std_tool)
                else:
                    raise ValueError(f"Tool '{tool}' not found in tools")
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

    def filter_tools_by_tool_choice(self, tools: list[Tool], tool_choice: str) -> list[Tool]:
        if tool_choice == "auto" or tool_choice == "required":
            return tools
        if tool_choice == "none":
            return []
        return [tool for tool in tools if tool.name == tool_choice]

    
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
        elif tool_choice != 'none':
            print(f"Tools NOT called and tool_choice={tool_choice} NOT OBEYED")
            return False 
        
        print(f"tool_choice={tool_choice} OBEYED")
        return True       


    def do_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        for tool_call in tool_calls:
            if tool_callable := self.tool_callables.get(tool_call.tool_name):
                # TODO catch ToolCallError
                tool_call.output = tool_callable(**(tool_call.arguments or {}))
            else:
                raise ValueError(f"Tool '{tool_call.tool_name}' callable not found")
        
        return tool_calls

    # Chat
    def chat(
            self,
            messages: Sequence[Union[Message, str, dict[str, Any]]],
            provider: Optional[AIProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,             
            tools: Optional[Sequence[ToolInput]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,

            return_on: Optional[Union[Literal["content", "tool_call", "message"], str, Collection[str]]] = "content",
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
        # client_messages = [client.prep_input_message(message) for message in std_messages]
        client_messages, system_prompt = client.prep_input_messages_and_system_prompt(std_messages, system_prompt)

        if tools:
            std_tools = self.standardize_tools(tools)
            client_tools = [client.prep_input_tool(tool) for tool in std_tools]
        else:
            std_tools = client_tools = None

        if tool_choice:
            if isinstance(tool_choice, Sequence) and not isinstance(tool_choice, str):
                std_tool_choice = self.standardize_tool_choice(tool_choice[0])
                std_tool_choice_queue = [self.standardize_tool_choice(tool_choice) for tool_choice in tool_choice[1:]]
            else:
                std_tool_choice = self.standardize_tool_choice(tool_choice)
                std_tool_choice_queue = None
            client_tool_choice = client.prep_input_tool_choice(std_tool_choice)
        else:
            std_tool_choice = std_tool_choice_queue = client_tool_choice = None

        # if tools:
        #     std_tools = self.standardize_tools(tools)
        #     # client_tools = [client.prep_input_tool(tool) for tool in std_tools]
        #     std_tools_filtered = self.filter_tools_by_tool_choice(std_tools, std_tool_choice) if std_tool_choice else std_tools
        #     client_tools = [client.prep_input_tool(tool) for tool in std_tools_filtered]
        # else:
        #     std_tools = client_tools = None

        if response_format:
            client_response_format = client.prep_input_response_format(response_format)
        else:
            client_response_format = None
        
        # Chat and ToolCall handling loop
        while (response := client.chat(
                messages=client_messages,                 
                model=model, 
                system_prompt=system_prompt,
                tools=client_tools, 
                tool_choice=client_tool_choice,
                response_format=client_response_format, 
                **kwargs
            )):

            # TODO check if response is an APIError

            std_message, client_message = client.extract_std_and_client_messages(response)
            print("std_message:", std_message)

            # Enforce Tool Choice: Check if tool choice is obeyed
            # auto:  
            if enforce_tool_choice and std_tool_choice != 'auto' and std_tool_choice is not None:
                if self.check_tool_choice_obeyed(std_tool_choice, std_message.tool_calls):
                    # TODO implement tool choice sequence
                    if std_tool_choice_queue and std_tools:
                        # Update std_tools and client_tools with next tool choice
                        std_tool_choice = std_tool_choice_queue.pop(0)
                        client_tool_choice = client.prep_input_tool_choice(std_tool_choice)

                        # # # Filter tools by tool choice            
                        # std_tools_filtered = self.filter_tools_by_tool_choice(std_tools, std_tool_choice)
                        # client_tools = [client.prep_input_tool(tool) for tool in std_tools_filtered]
                    else:
                        std_tool_choice = client_tool_choice = None # set to auto for next iteration
                        
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
                    return_on == "tool_call" # return_on is a string "tool_call"
                    or tool_call.tool_name == return_on # return_on is a tool name
                    or (isinstance(return_on, Collection) and tool_call.tool_name in return_on) # return_on is a collection of tool names
                    for tool_call in tool_calls):
                    print("returning on tool call", return_on)
                    break

                tool_calls = self.do_tool_calls(tool_calls)
                for std_tool_message in client.split_tool_call_outputs_into_messages(tool_calls):
                    std_messages.append(std_tool_message)
                    client_messages.append(client.prep_input_message(std_tool_message))

                # for tool_call in tool_calls:
                #     std_tool_output_message = self.do_tool_call(tool_call)
                #     client_tool_output_message = client.prep_input_message(std_tool_output_message)
                #     # Update messages with tool outputs
                #     std_messages.append(std_tool_output_message)
                #     client_messages.append(client_tool_output_message)
                
                # Process messages after submitting tool outputs
                continue

            print("Returning on content:", std_message.content)
            break

        return std_messages

    
    def evaluate(self, 
                 eval_type: str, 
                 content: Any, 
                 provider: Optional[AIProvider] = None,
                 model: Optional[str] = None,
                 **kwargs
                 ) -> Any:
        
        if (eval_type_parameters := self.eval_types.get(eval_type)) is None:
            raise ValueError(f"Eval type '{eval_type}' not found in eval_types")

        if isinstance(eval_type_parameters.tool_choice, str):
            last_tool_choice = eval_type_parameters.tool_choice
        elif eval_type_parameters.tool_choice:
            last_tool_choice = eval_type_parameters.tool_choice[-1]
        else:
            last_tool_choice = None

        return_on = eval_type_parameters.return_on or last_tool_choice or "content"

        input_messages = make_few_shot_prompt(
            system_prompt=eval_type_parameters.system_prompt,
            examples=eval_type_parameters.examples,
            content=content
        )

        messages = self.chat(
            messages=input_messages,
            provider=provider,
            model=model,
            tools=eval_type_parameters.tools,
            tool_choice=eval_type_parameters.tool_choice,
            response_format=eval_type_parameters.response_format,
            return_on=return_on,
            **kwargs
        )
        
        return_as = eval_type_parameters.return_as
        if return_as == "messages":
            return messages
        
        last_message = messages[-1]
        if return_as == "last_message":
            return last_message
        
        if last_message.content:
            return last_message.content
        
        if last_message.tool_calls:                
            if return_as == "last_content_or_all_tool_call_args":
                return [tool_call.arguments for tool_call in last_message.tool_calls]
            
            # default return_as == "last_content_or_tool_call_args"
            return last_message.tool_calls[-1].arguments
        
        raise ValueError("No content or tool call arguments to return")
            



from typing import Optional, Union, Any, Literal, Mapping, Type, Callable, Collection, Sequence, Type, List, Dict, Tuple, Self, Iterable

from json import dumps as json_dumps
from .baseaiclientwrapper import BaseAIClientWrapper

from ._types import Message, Tool, ToolCall, EvaluateParameters, EvaluateParametersInput, ToolInput, MessageInput
from ._convert_types import tool_from_dict, stringify_content, make_few_shot_prompt, standardize_eval_prameters, standardize_messages, standardize_tools, standardize_tool_choice

AIProvider = Literal["anthropic", "openai", "ollama"]

# ToolInput = Union[Tool, dict[str, Any], str]
# EvaluateParametersInput = Union[EvaluateParameters, dict[str, Any]]

class UnifAIClient:
    TOOLS: list[ToolInput] = []
    TOOL_CALLABLES: dict[str, Callable] = {}
    EVAL_PARAMETERS: list[EvaluateParametersInput] = []


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


    def filter_tools_by_tool_choice(self, tools: list[Tool], tool_choice: str) -> list[Tool]:
        if tool_choice == "auto" or tool_choice == "required":
            return tools
        if tool_choice == "none":
            return []
        return [tool for tool in tools if tool.name == tool_choice]

    
    def chat(
            self,
            messages: Optional[Sequence[Union[Message, str, dict[str, Any]]]] = None,
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
                tool_choice_error_retries=tool_choice_error_retries
            )
            if messages:
                chat.run()
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
        
              
class Chat:

    def __init__(self, 
                 parent: UnifAIClient,
                 provider: AIProvider,    
                 messages: Sequence[Union[Message, str, dict[str, Any]]],                         
                 model: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 tools: Optional[Sequence[ToolInput]] = None,
                 tool_callables: Optional[dict[str, Callable]] = None,
                 tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,                 
                 response_format: Optional[Union[str, dict[str, str]]] = None,
                 
                 return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
                 enforce_tool_choice: bool = False,
                 tool_choice_error_retries: int = 3,
                 ):
        

        self.parent = parent
        self.provider = provider
        self.set_provider(provider)
        self.set_model(model)
        self.set_system_prompt(system_prompt)
        self.set_messages(messages, system_prompt)
        self.set_tools(tools)
        self.set_tool_callables(tool_callables)
        self.set_tool_choice(tool_choice)
        self.set_response_format(response_format)
        self.return_on = return_on
        self.enforce_tool_choice = enforce_tool_choice
        self.tool_choice_error_retries = tool_choice_error_retries

    def set_provider(self, provider: AIProvider) -> Self:        
        self.client = self.parent.get_client(provider)
        if provider != self.provider:
            pass
            # Need reformat args?
        self.provider = provider
        return self
    
    def set_model(self, model: Optional[str]) -> Self:
        self.model = model or self.client.default_model
        return self
    
    def set_system_prompt(self, system_prompt: Optional[str]) -> Self:
        self.system_prompt = system_prompt
        return self    
    
    def set_messages(self, 
                     messages: Sequence[Union[Message, str, dict[str, Any]]],
                     system_prompt: Optional[str] = None) -> Self:

        # standardize inputs and prep copies for client in its format
        # (Note to self: 2 copies are stored to prevent converting back and forth between formats on each iteration.
        # this choice is at the cost of memory but is done to prevent unnecessary conversions 
        # and allow for easier debugging and error handling.
        # May revert back to optimizing for memory later if needed.)
        self.std_messages = standardize_messages(messages)
        self.client_messages, self.system_prompt = self.client.prep_input_messages_and_system_prompt(
            self.std_messages, system_prompt or self.system_prompt)
        return self
    
    def extend_messages(self, std_messages: Iterable[Message]) -> None:
        for std_message in std_messages:
            self.std_messages.append(std_message)
            self.client_messages.append(self.client.prep_input_message(std_message))  

    
    def set_tools(self, tools: Optional[Sequence[ToolInput]]) -> Self:
        if tools:
            self.std_tools = standardize_tools(tools, tool_dict=self.parent.tools)
            self.client_tools = [self.client.prep_input_tool(tool) for tool in self.std_tools.values()]
        else:
            self.std_tools = self.client_tools = None
        return self
    
    def set_tool_callables(self, tool_callables: Optional[dict[str, Callable]]) -> Self:
        if tool_callables:
            self.tool_callables = {**self.parent.tool_callables, **tool_callables}
        else:
            self.tool_callables = self.parent.tool_callables
        return self
    
    def set_tool_choice(self, tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]]) -> Self:
        if tool_choice:
            if isinstance(tool_choice, Sequence) and not isinstance(tool_choice, str):
                self.std_tool_choice = standardize_tool_choice(tool_choice[0])
                self.std_tool_choice_queue = [standardize_tool_choice(tool_choice) for tool_choice in tool_choice[1:]]
            else:
                self.std_tool_choice = standardize_tool_choice(tool_choice)
                self.std_tool_choice_queue = None
            self.client_tool_choice = self.client.prep_input_tool_choice(self.std_tool_choice)
        else:
            self.std_tool_choice = self.std_tool_choice_queue = self.client_tool_choice = None
        return self
    
    def set_response_format(self, response_format: Optional[Union[str, dict[str, str]]]) -> Self:
        if response_format:
            self.client_response_format = self.client.prep_input_response_format(response_format)
        else:
            self.client_response_format = None
        return self
    

    def enforce_tool_choice_needed(self) -> bool:
        return self.enforce_tool_choice and self.std_tool_choice != 'auto' and self.std_tool_choice is not None    
    
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
    
    def handle_tool_choice_obeyed(self, std_message: Message) -> None:
        if self.std_tool_choice_queue and self.std_tools:
            # Update std_tools and client_tools with next tool choice
            self.std_tool_choice = self.std_tool_choice_queue.pop(0)
            self.client_tool_choice = self.client.prep_input_tool_choice(self.std_tool_choice)
        else:
            self.std_tool_choice = self.client_tool_choice = None

    def handle_tool_choice_not_obeyed(self, std_message: Message) -> None:
        self.tool_choice_error_retries -= 1

    def check_return_on_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        if self.return_on == "tool_call":
            # return on is string "tool_call" = return on any tool call
            return True
        if isinstance(self.return_on, Collection):
            # return on is a collection of tool names. True if any tool call name is in the return_on collection
            return any(tool_call.tool_name in self.return_on for tool_call in tool_calls)
        
        # return on is a tool name. True if any tool call name is the same as the return_on tool name
        return any(tool_call.tool_name == self.return_on for tool_call in tool_calls)   


    def do_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            if self.std_tools and (tool := self.std_tools.get(tool_name)) and tool.callable:
                tool_callable = tool.callable
            elif self_tool_callable := self.tool_callables.get(tool_name):
                tool_callable = self_tool_callable            
            elif parent_tool_callable := self.parent.tool_callables.get(tool_name):                
                tool_callable = parent_tool_callable
            else:
                raise ValueError(f"Tool '{tool_name}' callable not found")
            
            # TODO catch ToolCallError
            tool_call.output = tool_callable(**tool_call.arguments)
                   
        return tool_calls
                    
                        
    def extend_messages_with_tool_outputs(self, 
                                        tool_calls: Sequence[ToolCall],
                                        content: Optional[str] = None
                                        ) -> None:
        # std_tool_messages = self.client.split_tool_call_outputs_into_messages(tool_calls, content)
        # self.extend_messages(std_tool_messages)
        self.extend_messages(self.client.split_tool_outputs_into_messages(tool_calls, content))
             


    def run(self) -> Self:
        while True:
            response = self.client.chat(
                messages=self.client_messages,                 
                model=self.model, 
                system_prompt=self.system_prompt,
                tools=self.client_tools, 
                tool_choice=self.client_tool_choice,
                response_format=self.client_response_format
            )
            # TODO check if response is an APIError

            std_message, client_message = self.client.extract_output_assistant_messages(response)
            print("std_message:", std_message)

            # Enforce Tool Choice: Check if tool choice is obeyed
            # auto:  
            if self.enforce_tool_choice and self.std_tool_choice != 'auto' and self.std_tool_choice is not None:
                if self.check_tool_choice_obeyed(self.std_tool_choice, std_message.tool_calls):
                    self.handle_tool_choice_obeyed(std_message)
                elif self.tool_choice_error_retries > 0:
                    self.handle_tool_choice_not_obeyed(std_message)
                    # Continue to next iteration without updating messages (retry)
                    continue
                else:
                    print("Tool choice error retries exceeded")
                    raise ValueError("Tool choice error retries exceeded")
                
            # Update messages with assistant message
            self.std_messages.append(std_message)
            self.client_messages.append(client_message)

            if self.return_on == "message":
                print("returning on message")
                break

            if tool_calls := std_message.tool_calls:
                # Return on tool_call before processing messages
                if self.check_return_on_tool_call(tool_calls):
                    break

                tool_calls = self.do_tool_calls(tool_calls)
                self.extend_messages_with_tool_outputs(tool_calls)
                # Process messages after submitting tool outputs
                continue

            print("Returning on content:", std_message.content)
            break

        return self


    @property
    def messages(self) -> list[Message]:
        return self.std_messages
    
    @property
    def last_message(self) -> Optional[Message]:
        if self.std_messages:
            return self.std_messages[-1]
    
    @property
    def last_content(self) -> Optional[str]:
        if last_message := self.last_message:
            return last_message.content

    # alias for last_content
    content = last_content
    
    @property
    def last_tool_calls(self) -> Optional[list[ToolCall]]:
        if last_message := self.last_message:
            return last_message.tool_calls
        
    # alias for last_tool_calls
    tool_calls = last_tool_calls

    @property
    def last_tool_calls_args(self) -> Optional[list[Mapping[str, Any]]]:
        if last_tool_calls := self.last_tool_calls:
            return [tool_call.arguments for tool_call in last_tool_calls]


    @property
    def last_tool_call(self) -> Optional[ToolCall]:
        if last_tool_calls := self.last_tool_calls:
            return last_tool_calls[-1]
        
            
    @property
    def last_tool_call_args(self) -> Optional[Mapping[str, Any]]:
        if last_tool_call := self.last_tool_call:
            return last_tool_call.arguments
        

    def send_message(self, *message: Union[Message, str, dict[str, Any]]) -> Message:
        if not message:
            raise ValueError("No message(s) provided")

        messages = standardize_messages(message)

        # prevent error when using multiple return_tools without submitting tool outputs
        if (last_message := self.last_message) and last_message.role == "assistant" and last_message.tool_calls:
            # Submit tool outputs before sending new messages. 
            # Use first new message content as content of tool message or send after as user message based on provider
            self.extend_messages_with_tool_outputs(last_message.tool_calls, content=messages.pop(0).content)
        
        self.extend_messages(messages)
        return self.run().last_message
    
    
    def submit_tool_outputs(self, 
                            tool_calls: Sequence[ToolCall], 
                            tool_outputs: Optional[Sequence[Any]]
                            ) -> Self:
        if tool_outputs:
            for tool_call, tool_output in zip(tool_calls, tool_outputs):
                tool_call.output = tool_output
        self.extend_messages_with_tool_outputs(tool_calls)
        return self.run()
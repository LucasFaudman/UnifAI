from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai.types import (
    LLMProvider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    ToolCall,
    Usage,
)
from unifai.type_conversions import standardize_tools, standardize_messages, standardize_tool_choice, standardize_response_format

class Chat:

    def __init__(self, 
                 parent,
                 provider: LLMProvider,    
                 messages: Sequence[MessageInput],                         
                 model: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 tools: Optional[Sequence[ToolInput]] = None,
                 tool_callables: Optional[dict[str, Callable]] = None,
                 tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,                 
                 response_format: Optional[Union[Literal["text", "json", "json_schema"], dict[Literal["type"], Literal["text", "json", "json_schema"]]]] = None,
                 
                 return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
                 enforce_tool_choice: bool = False,
                 tool_choice_error_retries: int = 3,

                 max_tokens: Optional[int] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 seed: Optional[int] = None,
                 stop_sequences: Optional[list[str]] = None, 
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,                   
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

        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop_sequences = stop_sequences
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.rejected_messages = []
        self.usage = Usage(input_tokens=0, output_tokens=0)


    def set_provider(self, provider: LLMProvider, model: Optional[str]=None) -> Self:        
        self.client = self.parent.get_client(provider)
        if provider != self.provider:
            self.reformat_client_args()
        self.provider = provider
        self.set_model(model)
        return self
    
    def reformat_client_args(self) -> Self:
        self.client_messages, self.system_prompt = self.client.prep_input_messages_and_system_prompt(
            self.std_messages,self.system_prompt)
        if self.std_tools:
            self.client_tools = [self.client.prep_input_tool(tool) for tool in self.std_tools.values()]
        if self.std_tool_choice:
            self.client_tool_choice = self.client.prep_input_tool_choice(self.std_tool_choice)
        if self.std_response_format:
            self.client_response_format = self.client.prep_input_response_format(self.std_response_format)
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
                self.std_tool_choice_queue = [standardize_tool_choice(tool_choice) for tool_choice in tool_choice]                
            else:
                self.std_tool_choice_queue = [standardize_tool_choice(tool_choice)]

            self.std_tool_choice_index = 0
            self.std_tool_choice = self.std_tool_choice_queue[self.std_tool_choice_index]
            self.client_tool_choice = self.client.prep_input_tool_choice(self.std_tool_choice)
        else:
            self.std_tool_choice = self.std_tool_choice_queue = self.std_tool_choice_index = self.client_tool_choice = None
        return self
    

    def set_response_format(self, response_format: Optional[Union[Literal["text", "json", "json_schema"], dict[Literal["type"], Literal["text", "json", "json_schema"]]]]) -> Self:
        if response_format:
            self.std_response_format = standardize_response_format(response_format)
            self.client_response_format = self.client.prep_input_response_format(self.std_response_format)
        else:
            self.std_response_format = self.client_response_format = None
        return self
    

    def enforce_tool_choice_needed(self) -> bool:
        return self.enforce_tool_choice and self.std_tool_choice != 'auto' and self.std_tool_choice is not None    
    

    def check_tool_choice_obeyed(self, tool_choice: str, tool_calls: Optional[list[ToolCall]]) -> bool:
        if tool_choice == "auto":
            print("tool_choice='auto' OBEYED")
            return True
        
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
            if self.std_tool_choice_index is not None and self.std_tool_choice_index + 1 < len(self.std_tool_choice_queue):
                self.std_tool_choice_index += 1
            else:
                self.std_tool_choice_index = 0
            self.std_tool_choice = self.std_tool_choice_queue[self.std_tool_choice_index]
            self.client_tool_choice = self.client.prep_input_tool_choice(self.std_tool_choice)
        else:
            self.std_tool_choice = self.client_tool_choice = None


    def handle_tool_choice_not_obeyed(self, std_message: Message) -> None:
        self.tool_choice_error_retries -= 1
        self.rejected_messages.append(std_message)


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
                                        # content: Optional[str] = None
                                        ) -> None:

        # tool_message = Message(role="tool", tool_calls=tool_calls, content=content)
        tool_message = Message(role="tool", tool_calls=tool_calls)
        self.std_messages.append(tool_message)
        self.client_messages.extend(map(self.client.prep_input_message, self.client.split_tool_message(tool_message)))
        # self.extend_messages(self.client.split_tool_outputs_into_messages(tool_calls, content))
             

    def client_kwargs(self, override_kwargs: dict) -> dict[str, Any]:             
        return dict(
                    messages=self.client_messages,                 
                    model=self.model, 
                    system_prompt=self.system_prompt,
                    tools=self.client_tools, 
                    tool_choice=self.client_tool_choice,
                    response_format=self.client_response_format,
                    max_tokens=self.max_tokens,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    seed=self.seed,
                    stop_sequences=self.stop_sequences,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,                    
                    **override_kwargs
            )
    
    def run(self, **kwargs) -> Self:
        while True:
            std_message, client_message = self.client.chat(
                **self.client_kwargs(override_kwargs=kwargs)
            )
            # TODO handle error messages
            print("\nstd_message:", std_message)

            # Update usage for entire chat
            self.usage += std_message.response_info.usage

            # Enforce Tool Choice: Check if tool choice is obeyed
            # auto:  
            if self.enforce_tool_choice and self.std_tool_choice is not None:
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
    
    def run_stream(self, **kwargs) -> Generator[MessageChunk, None, Self]:
        while True:
            # print(f"{self.client_kwargs(stream=True, **kwargs)=}")
            print(f"{self.client_kwargs(kwargs)['tool_choice']=}")
            std_message, client_message = yield from self.client.chat_stream(
               **self.client_kwargs(override_kwargs=kwargs)
            )
            # TODO handle error messages
            # print("\nstd_message:", std_message)

            # Update usage for entire chat
            self.usage += std_message.response_info.usage

            # Enforce Tool Choice: Check if tool choice is obeyed
            # auto:  
            if self.enforce_tool_choice and self.std_tool_choice is not None:
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


    def clear_messages(self) -> Self:
        self.std_messages = []
        self.client_messages = []
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

    def _send_message(self, *message: Union[Message, str, dict[str, Any]], **kwargs):
        if not message:
            raise ValueError("No message(s) provided")

        messages = standardize_messages(message)

        # prevent error when using multiple return_tools without submitting tool outputs
        if (last_message := self.last_message) and last_message.role == "assistant" and last_message.tool_calls:
            # Submit tool outputs before sending new messages. 
            # Use first new message content as content of tool message or send after as user message based on provider
            # self.extend_messages_with_tool_outputs(last_message.tool_calls, content=messages.pop(0).content)
            self.extend_messages_with_tool_outputs(last_message.tool_calls)

        
        self.extend_messages(messages)
        

    def send_message(self, *message: Union[Message, str, dict[str, Any]], **kwargs) -> Message:
        self._send_message(*message, **kwargs)
        self.run(**kwargs)
        return self.last_message
    
    
    def send_message_stream(self, *message: Union[Message, str, dict[str, Any]], **kwargs) -> Generator[MessageChunk, None, Message]:
        self._send_message(*message, **kwargs)
        yield from self.run_stream(**kwargs)
        return self.last_message


    
    def _submit_tool_outputs(self, 
                            tool_calls: Sequence[ToolCall], 
                            tool_outputs: Optional[Sequence[Any]],
                            **kwargs
                            ):
        if tool_outputs:
            for tool_call, tool_output in zip(tool_calls, tool_outputs):
                tool_call.output = tool_output
        self.extend_messages_with_tool_outputs(tool_calls)
        

    def submit_tool_outputs(self,
                            tool_calls: Sequence[ToolCall], 
                            tool_outputs: Optional[Sequence[Any]],
                            **kwargs
                            ) -> Self:
        self._submit_tool_outputs(tool_calls, tool_outputs, **kwargs)
        return self.run(**kwargs)
    

    def submit_tool_outputs_stream(self,
                            tool_calls: Sequence[ToolCall], 
                            tool_outputs: Optional[Sequence[Any]],
                            **kwargs
                            ) -> Generator[MessageChunk, None, Self]:
        self._submit_tool_outputs(tool_calls, tool_outputs, **kwargs)
        yield from self.run_stream(**kwargs)
        return self




    # def send_message(self, *message: Union[Message, str, dict[str, Any]], **kwargs) -> Message:
    #     if not message:
    #         raise ValueError("No message(s) provided")

    #     messages = standardize_messages(message)

    #     # prevent error when using multiple return_tools without submitting tool outputs
    #     if (last_message := self.last_message) and last_message.role == "assistant" and last_message.tool_calls:
    #         # Submit tool outputs before sending new messages. 
    #         # Use first new message content as content of tool message or send after as user message based on provider
    #         self.extend_messages_with_tool_outputs(last_message.tool_calls, content=messages.pop(0).content)
        
    #     self.extend_messages(messages)
    #     return self.run(**kwargs).last_message
    
    
    # def submit_tool_outputs(self, 
    #                         tool_calls: Sequence[ToolCall], 
    #                         tool_outputs: Optional[Sequence[Any]],
    #                         **kwargs
    #                         ) -> Self:
    #     if tool_outputs:
    #         for tool_call, tool_output in zip(tool_calls, tool_outputs):
    #             tool_call.output = tool_output
    #     self.extend_messages_with_tool_outputs(tool_calls)
    #     return self.run(**kwargs)
    

    def __str__(self) -> str:
        return f"Chat(provider={self.provider}, model={self.model},  messages={len(self.std_messages)}, tools={len(self.std_tools) if self.std_tools else None}, tool_choice={self.std_tool_choice}, response_format={self.std_response_format})"
from typing import TYPE_CHECKING, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator, TypeVar, Generic
from copy import deepcopy


# if TYPE_CHECKING:
from ..components._base_components._base_tool_caller import ToolCaller
from ..components._base_components._base_tokenizer import Tokenizer
from ..components.prompt_template import PromptTemplate
from ..configs.llm_config import LLMConfig
from ..configs.tool_caller_config import ToolCallerConfig
from ..configs.tokenizer_config import TokenizerConfig

from ._base_components._base_llm import LLM
from ..components._base_components._base_component import UnifAIComponent
from ..configs.chat_config import ChatConfig


from ..types.annotations import ComponentName, ModelName, ProviderName, ToolName, ToolInput, BaseModel
from ..types import (
    ProviderName, 
    Message,
    MessageChunk,
    MessageInput, 
    ResponseFormatInput,
    ReturnOnInput,
    Tool,
    ToolInput,
    ToolCall,
    ToolChoiceInput,
    ToolChoice,
    ToolName,
    ResponseInfo,
    Usage,
)
from ..type_conversions import standardize_tool, standardize_tools, standardize_messages, standardize_message, standardize_tool_choice, standardize_response_format
from ..utils import combine_dicts, stringify_content

ChatConfigT = TypeVar("ChatConfigT", bound=ChatConfig)

class BaseChat(UnifAIComponent[ChatConfigT], Generic[ChatConfigT]):
    component_type = "chat"
    provider = "base"    
    can_get_components = True

    def _setup(self) -> None:
        # Setup runtime variables to allow for dynamic setting of components and access by subclasses        
        # but wait for full initialization when run for the first time before initializing components and runtime variables
        # this way chats that are initialized but not run can be created and passed around 
        # without overhead until the chat is actually run and the resources need to be allocated        
        self._llm = None
        self._llm_model = None
        self._system_prompt = None

        self._unifai_examples = []
        self._client_examples = []

        self._unifai_messages = []
        self._client_messages = []
        self.deleted_messages = []
        self.history = []
        
        self._unifai_response_format = None
        self._client_response_format = None

        self._unifai_tools = None
        self._client_tools = None

        self._unifai_tool_choice = None
        self._client_tool_choice = None

        self._tool_choice_queue = None
        self._tool_choice_index = 0
        self._tool_caller = None
        self._tool_callables = None

        self._return_on = "content"        
        self.usage = Usage(input_tokens=0, output_tokens=0)

        # runtime variables optionally set by init_kwargs during initialization
        if (messages := self.init_kwargs.get("messages", None)) is not None:
            self.messages = messages
        else:
            self.messages = []
        self.system_prompt_kwargs = self.init_kwargs.get("system_prompt_kwargs", {})
        self.tool_registry = self.init_kwargs.get("tool_registry", {})

        # Wait until run to fully initialize components and runtime variables from config
        self._fully_initialized = False

    def _init_config_components(self) -> None:
        self.llm = self.config.llm
        self.llm_model = self.config.llm_model
        self.system_prompt = self.config.system_prompt
        self.examples = self.config.examples

        self.tools = self.config.tools
        self.tool_callables = self.config.tool_callables 
        self.tool_caller = self.config.tool_caller
        self.tool_choice = self.config.tool_choice
        self.tool_choice_error_retries = self.config.tool_choice_error_retries
        self.enforce_tool_choice = self.config.enforce_tool_choice

        self.response_format = self.config.response_format
        self.return_on = self.config.return_on

        self._fully_initialized = True

    def _init_if_needed(self) -> None:
        if not self._fully_initialized:
            self._init_config_components()


    def _get_llm_client(self, llm: "ProviderName | LLMConfig | tuple[ProviderName, ComponentName]") -> "LLM":
        return self._get_component("llm", llm)
    
    def _get_tokenizer(self, tokenizer: "ProviderName | TokenizerConfig | tuple[ProviderName, ComponentName]") -> "Tokenizer":
        return self._get_component("tokenizer", tokenizer)    

    def _get_tool_caller(self, tool_caller: "ProviderName | ToolCallerConfig | tuple[ProviderName, ComponentName]") -> "ToolCaller":
        return self._get_component("tool_caller", tool_caller)

    @property
    def llm(self) -> "LLM":
        if self._llm is None:
            self._llm = self._get_llm_client(self.config.llm)
        return self._llm
    
    @llm.setter
    def llm(self, llm: "LLM | ProviderName | LLMConfig | tuple[ProviderName, ComponentName]") -> None:
        self._llm_model = None # reset model when llm changes
        _old_llm_provider = self._llm.provider if self._llm else None
        if isinstance(llm, LLM):
            self._llm = llm
        else:
            self._llm = self._get_llm_client(llm)
        if _old_llm_provider and _old_llm_provider != self._llm.provider:
            self.reformat_client_inputs()

    def reformat_client_inputs(self) -> Self:
        if self._unifai_messages:
            self._client_messages.clear()
            self._client_messages.extend(self.llm.format_messages(self._unifai_messages))            
        if self._unifai_tools:
            self._client_tools = list(map(self.llm.format_tool, self._unifai_tools.values()))
        if self._unifai_tool_choice:
            self._client_tool_choice = self.llm.format_tool_choice(self._unifai_tool_choice)
        if self._unifai_response_format:
            self._client_response_format = self.llm.format_response_format(self._unifai_response_format)
        return self
        
    @property
    def llm_provider(self) -> ProviderName:
        return self.llm.provider
    
    @llm_provider.setter
    def llm_provider(self, provider: ProviderName) -> None:
        self.llm = provider

    @property
    def llm_model(self) -> ModelName:
        if self._llm_model is None:
            self._llm_model = self.config.llm_model or self.llm.default_model
        return self._llm_model
    
    @llm_model.setter
    def llm_model(self, model: Optional[ModelName]) -> None:
        self._llm_model = model

    @property
    def system_prompt(self) -> str|None:
        # None or empty string
        if not (_system_prompt := self._system_prompt):
            return None
        # string and nothing to format
        if (_is_str := isinstance(_system_prompt, str)) and not self.system_prompt_kwargs:
            return _system_prompt
        # format string or PromptTemplate with system_prompt_kwargs
        if _is_str or isinstance(_system_prompt, PromptTemplate):
            return _system_prompt.format(**self.system_prompt_kwargs)
        # call system_prompt function with system_prompt_kwargs
        return _system_prompt(**self.system_prompt_kwargs)
        
    @system_prompt.setter
    def system_prompt(self, system_prompt: Optional[str | PromptTemplate | Callable[...,str]]) -> None:
        self._system_prompt = system_prompt

    @property
    def examples(self) -> Optional[list[Message]]:
        return self._unifai_examples
    
    @examples.setter
    def examples(self, examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]]) -> None:
        if self._unifai_examples:
            self._unifai_examples.clear()
        if self._client_examples:
            self._client_examples.clear()
        if not examples:
            return        
        for example in examples:
            if isinstance(example, Message):
                self._unifai_examples.append(example)
            else:
                self._unifai_examples.append(Message(role="user", content=stringify_content(example['input'])))
                self._unifai_examples.append(Message(role="assistant", content=stringify_content(example['response'])))
        self._client_examples.extend(self.llm.format_messages(self._unifai_examples))

    @property
    def messages(self) -> list[Message]:
        return self._unifai_messages

    @messages.setter
    def messages(self, messages: Sequence[MessageInput]) -> None: 
        if self._client_messages:
            self._client_messages.clear()        
        if self._unifai_messages:
            self.deleted_messages.extend(self._unifai_messages) # delete any existing messages
            self._unifai_messages.clear() # preserve original list in case of outside references
        self.extend_messages(messages)

    def append_message(self, message: MessageInput) -> None:
        self.extend_messages([message])

    def extend_messages(self, messages: Sequence[MessageInput]) -> None:
        # convert inputs to UnifAI format
        unifai_messages = standardize_messages(messages)

        # set system prompt if first message is a system message and pop it before formatting client messages
        # (Note: system prompt and examples are not included in client messages since they are not always part of the conversation)
        if unifai_messages and unifai_messages[0].role == "system":
            self.system_prompt = unifai_messages.pop(0).content

        # standardize inputs and prep copies for client in its format
        # (Note to self: 2 copies are stored to prevent converting back and forth between formats on each iteration.
        # this choice is at the cost of memory but is done to prevent unnecessary conversions 
        # and allow for easier debugging and error handling.
        # May revert back to optimizing for memory later if needed.)
        self._unifai_messages.extend(unifai_messages)       
        # format messages for client
        self._client_messages.extend(self.llm.format_messages(unifai_messages))
        # update history with new messages
        self.history.extend(unifai_messages)

    def extend_messages_with_tool_outputs(self, tool_calls: list[ToolCall], content: Optional[str] = None) -> None:
        self.append_message(Message(role="tool", tool_calls=tool_calls, content=content))
        
    def clear_messages(self) -> Self:
        self.deleted_messages.extend(self._unifai_messages)
        self._unifai_messages.clear()
        self._client_messages.clear()
        return self
    
    def pop_message(self, index: int = -1) -> Message:
        self._client_messages.pop(index)
        unifai_message = self._unifai_messages.pop()
        self.deleted_messages.append(unifai_message)
        return unifai_message

    @property
    def last_message(self) -> Optional[Message]:
        if self._unifai_messages:
            return self._unifai_messages[-1]
    
    @property
    def tools(self) -> dict[ToolName, Tool]|None:
        return self._unifai_tools

    @tools.setter
    def tools(self, tools: Optional[list[ToolInput]|dict[ToolName, Tool]]):
        if tools:
            self._unifai_tools = standardize_tools(tools, self.tool_registry) if not isinstance(tools, dict) else tools
            self._client_tools = list(map(self.llm.format_tool, self._unifai_tools.values()))
        else:
            self._unifai_tools = self._client_tools = None

    def add_tool(self, tool: ToolInput) -> None:
        tool = standardize_tool(tool, self.tool_registry)
        if self._unifai_tools is None:
            self._unifai_tools = {}
        if self._client_tools is None:
            self._client_tools = []                        
        self._unifai_tools[tool.name] = tool
        self._client_tools.append(self.llm.format_tool(tool))
        
    @property
    def tool_callables(self) -> Optional[dict[str, Callable]]:
        return self._tool_callables

    @tool_callables.setter
    def tool_callables(self, tool_callables: Optional[dict[str, Callable]]) -> None:
        self._tool_callables = tool_callables
        if self._tool_caller and tool_callables is not None:
            self._tool_caller.set_tool_callables(self._resolve_tool_callables(tool_callables))

    def _resolve_tool_callables(self, tool_callables: Optional[dict[str, Callable]] = None) -> dict[str, Callable]:
        if self._unifai_tools:
            _callables_from_tools = {
                tool_name: tool.callable for tool_name, tool in self._unifai_tools.items() if tool.callable
            }
        else:
            _callables_from_tools = None
        if not (_callables_from_tools or tool_callables):
            return self._tool_callables or {}
        return combine_dicts(_callables_from_tools, self._tool_callables, tool_callables)            

    @property
    def tool_caller(self) -> Optional[ToolCaller]:
        return self._tool_caller

    @tool_caller.setter
    def tool_caller(
            self, 
            tool_caller: Optional[ToolCaller | ProviderName | ToolCallerConfig | tuple[ProviderName, ComponentName]]
    ) -> None:

        if tool_caller is None or isinstance(tool_caller, ToolCaller):
            self._tool_caller = tool_caller
        else:
            self._tool_caller = self._get_tool_caller(tool_caller)
            self._tool_caller.set_tool_callables(self._resolve_tool_callables())

    @property
    def tool_choice(self) -> Optional[ToolChoice]:
        return self._unifai_tool_choice
    
    @tool_choice.setter
    def tool_choice(self, tool_choice: Optional[ToolChoiceInput]):
        self._tool_choice_index = 0
        if tool_choice:
            if isinstance(tool_choice, str):
                tool_choice = [tool_choice]

            self._tool_choice_queue = list(map(standardize_tool_choice, tool_choice))            
            self._unifai_tool_choice = self._tool_choice_queue[self._tool_choice_index]
            self._client_tool_choice = self.llm.format_tool_choice(self._unifai_tool_choice)
        else:
            self._unifai_tool_choice = self._client_tool_choice = self._tool_choice_queue = None

    def enforce_tool_choice_needed(self) -> bool:
        return self.config.enforce_tool_choice and self._unifai_tool_choice != 'auto' and self._unifai_tool_choice is not None    
    
    def _check_tool_choice_obeyed(self, tool_choice: str, tool_calls: Optional[list[ToolCall]]) -> bool:
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
    
    def _handle_tool_choice_obeyed(self, message: Message) -> None:
        # reset retries
        self.tool_choice_error_retries = self.config.tool_choice_error_retries
        if self._tool_choice_queue and self._unifai_tools:
            if self._tool_choice_index + 1 < len(self._tool_choice_queue):
                # increment tool_choice_index to next choice in queue
                _new_index = self._tool_choice_index + 1
            else:                
                _new_index = 0 # reset to first choice in queue
            if self._tool_choice_index != _new_index: # optimize to prevent unnecessary reformatting when queue len is 1 
                self._tool_choice_index = _new_index
                self._unifai_tool_choice = self._tool_choice_queue[self._tool_choice_index]
                self._client_tool_choice = self.llm.format_tool_choice(self._unifai_tool_choice)
        else:
            # if had a tool_choice queue (and tools to choose from) before but now no longer have tools or a choice queue
            # reset tool_choice to None to prevent llm errors and allow for new tool_choice to be set on subsequent runs
            self._unifai_tool_choice = self._client_tool_choice = None

    def _handle_tool_choice_not_obeyed(self, message: Message) -> None:
        self.tool_choice_error_retries -= 1
        self.deleted_messages.append(message) # keep for calculating usage and debugging

    @property
    def return_on(self) -> Literal["content", "tool_call", "message"] | ToolName | list[ToolName]:
        return self._return_on
    
    @return_on.setter
    def return_on(self, return_on: Union[Literal["content", "tool_call", "message"], ToolName, Tool, list[ToolName | Tool]]) -> None:
        if return_on in ("content", "message"):            
            self._return_on = return_on
            return
        if isinstance(return_on, Tool):
            return_on = return_on.name
        if isinstance(return_on, str):
            if not self._unifai_tools:
                raise ValueError("No tools set to return on")
            if return_on != "tool_call" and return_on not in self._unifai_tools:
                raise ValueError(f"Tool '{return_on}' not found in tools")
            self._return_on = return_on
            return
        if isinstance(return_on, Collection):
            if not self._unifai_tools:
                raise ValueError("No tools set to return on")
            if not all(tool_name in self._unifai_tools for tool_name in return_on):
                raise ValueError("One or more tools not found in tools")
            self._return_on = [tool.name if isinstance(tool, Tool) else tool for tool in return_on]
            return
        raise ValueError(f"Invalid return_on: {return_on}")

    def _check_return_on_tool_call(self, tool_calls: list[ToolCall]) -> bool:
        # (type Literal['tool_call']) true on any tool call regardless of tool name
        if (_return_on := self._return_on) == "tool_call":
            return True
        # (type list[ToolName]) True if any of the called tools are in the return_on list
        if isinstance(_return_on, Collection):
            return any(tool_call.tool_name in _return_on for tool_call in tool_calls)        
        # (type ToolName) True if any of the called tools are the return_on tool
        return any(tool_call.tool_name == _return_on for tool_call in tool_calls)

    @property
    def response_format(self) -> Optional[str]:
        return self._unifai_response_format

    @response_format.setter
    def response_format(self, response_format: Optional[Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]]) -> None:
        if response_format:
            self._unifai_response_format = standardize_response_format(response_format)
            self._client_response_format = self.llm.format_response_format(self._unifai_response_format)
        else:
            self._unifai_response_format = self._client_response_format = None

    def _prep_chat_kwargs(self, override_kwargs: Optional[dict] = None) -> dict[str, Any]: 
        run_system_prompt = self.system_prompt
        run_client_messages = []
        if run_system_prompt is not None and self.llm._system_prompt_input_type == "first_message":
            run_client_messages.append(self.llm.format_system_message(Message(role="system", content=run_system_prompt)))
        if self._client_examples:
            run_client_messages.extend(self._client_examples)
        run_client_messages.extend(self._client_messages)

        chat_kwargs = dict(
                    messages=run_client_messages,                 
                    model=self.llm_model, 
                    system_prompt=self.system_prompt,
                    tools=self._client_tools, 
                    tool_choice=self._client_tool_choice,
                    response_format=self._client_response_format,
                    max_tokens=self.config.max_tokens_per_run, # TODO 
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    seed=self.config.seed,
                    stop_sequences=self.config.stop_sequences,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,                    
        )
        return combine_dicts(chat_kwargs, override_kwargs) if override_kwargs else chat_kwargs
    
    def _reset_run_counts(self) -> None:
        self._current_run_messages = 0
        self._current_run_tool_calls = 0
        self._current_run_tokens = 0
        self._current_run_input_tokens = 0
        self._current_run_output_tokens = 0
    
    def _handle_chat_response(self, message: Message, client_message: dict[str, Any]) -> bool:
        print(f"Assistant Response: {message}")
        self._current_run_messages += 1

        # Update usage for entire chat and current run
        if message.response_info and (usage := message.response_info.usage):
            self.usage += usage
            input_tokens = usage.input_tokens
            self._current_run_tokens += input_tokens
            self._current_run_input_tokens += input_tokens
            output_tokens = usage.output_tokens
            self._current_run_tokens += output_tokens
            self._current_run_output_tokens += output_tokens

        # Enforce Tool Choice: Check if tool choice is obeyed
        if self.enforce_tool_choice and self._unifai_tool_choice is not None:
            if self._check_tool_choice_obeyed(self._unifai_tool_choice, message.tool_calls):
                self._handle_tool_choice_obeyed(message) # Increment tool_choice_index to next choice in queue
            elif self.tool_choice_error_retries > 0:
                self._handle_tool_choice_not_obeyed(message)                
                return True # continue to next iteration without updating messages (retry)
            else:
                # TODO use UnifAIErrorType
                print("Tool choice error retries exceeded")
                raise ValueError("Tool choice error retries exceeded")
            
        # Update messages with assistant message
        self.history.append(message)
        self._unifai_messages.append(message)
        self._client_messages.append(client_message)

        if self.return_on == "message":
            print("returning on message")
            return False # break before processing tool_calls

        if tool_calls := message.tool_calls:
            self._current_run_tool_calls += len(tool_calls)

            # break before processing tool_calls when no tool_caller is set,
            # tool_call is in return_on list, is the return_on tool or is 'tool_call' (any tool_call)
            if not self.tool_caller or self._check_return_on_tool_call(tool_calls):
                return False # break before processing tool_calls

            # call tools and extend messages with tool outputs
            tool_calls = self.tool_caller.call_tools(tool_calls)
            self.extend_messages_with_tool_outputs(tool_calls)  
            return True # continue to next iteration after submitting tool outputs to process tool_callss
        
        return False # default to break on content

    def run(self, **kwargs) -> Self:
        self._init_if_needed()
        self._reset_run_counts()
        while self._current_run_messages < self.config.max_messages_per_run:      
            message, client_message = self.llm.chat(**self._prep_chat_kwargs(kwargs))
            if self._handle_chat_response(message, client_message):
                continue
            break
        return self
        
    def run_stream(self, **kwargs) -> Generator[MessageChunk, None, Self]:
        self._init_if_needed()
        self._reset_run_counts()
        while self._current_run_messages < self.config.max_messages_per_run:      
            message, client_message = yield from self.llm.chat_stream(**self._prep_chat_kwargs(kwargs))
            if self._handle_chat_response(message, client_message):
                continue
            break
        return self

    def _send_message(self, *message: MessageInput, **kwargs):
        if not message:
            raise ValueError("No message(s) provided")
        # prevent error when using multiple return_tools without submitting tool outputs
        while (last_message := self.last_message) and last_message.role == "assistant" and last_message.tool_calls:
            self.pop_message()                    
        self.extend_messages(message)
        
    def send_message(self, *message: MessageInput, **kwargs) -> Message|None:
        self._send_message(*message, **kwargs)
        self.run(**kwargs)
        return self.last_message
        
    def send_message_stream(self, *message: MessageInput, **kwargs) -> Generator[MessageChunk, None,  Message|None]:
        self._send_message(*message, **kwargs)
        yield from self.run_stream(**kwargs)
        return self.last_message
    
    def _submit_tool_outputs(self, 
                            tool_calls: list[ToolCall], 
                            tool_outputs: Optional[Iterable[Any]],
                            ):
        if tool_outputs:
            for tool_call, tool_output in zip(tool_calls, tool_outputs):
                tool_call.output = tool_output
        self.extend_messages_with_tool_outputs(tool_calls)
        
    def submit_tool_outputs(self,
                            tool_calls: list[ToolCall], 
                            tool_outputs: Optional[Iterable[Any]],
                            **kwargs
                            ) -> Self:
        self._submit_tool_outputs(tool_calls, tool_outputs)
        return self.run(**kwargs)
    
    def submit_tool_outputs_stream(self,
                            tool_calls: list[ToolCall], 
                            tool_outputs: Optional[Iterable[Any]],
                            **kwargs
                            ) -> Generator[MessageChunk, None, Self]:
        self._submit_tool_outputs(tool_calls, tool_outputs, **kwargs)
        yield from self.run_stream(**kwargs)
        return self

    # Convinience properties
    
    # Last message properties
    @property
    def last_deleted_message(self) -> Optional[Message]:
        if self.deleted_messages:
            return self.deleted_messages[-1]

    @property
    def last_content(self) -> Optional[str]:
        if last_message := (self.last_message or self.last_deleted_message):
            return last_message.content

    # alias for last_content
    content = last_content

    @property
    def last_response_info(self) -> Optional[ResponseInfo]:
        if last_message := (self.last_message or self.last_deleted_message):
            return last_message.response_info
        
    @property
    def last_usage(self) -> Optional[Usage]:
        if last_response_info := self.last_response_info:
            return last_response_info.usage
    
    @property
    def last_tool_calls(self) -> Optional[list[ToolCall]]:
        if last_message := (self.last_message or self.last_deleted_message):
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
        
    # History properties
    @property
    def all_messages(self) -> list[Message]:
        return self.history
    
    @property
    def all_contents(self) -> list[str|None]:
        return [message.content for message in self.history]
    
    @property
    def all_response_infos(self) -> list[ResponseInfo]:
        return [message.response_info for message in self.history if message.response_info]
    
    @property
    def all_usages(self) -> list[Usage]:
        return [response_info.usage for response_info in self.all_response_infos if response_info.usage]

    def __str__(self) -> str:
        return f"Chat(provider={self.llm_provider}, model={self.llm_model},  messages={len(self.messages)}, tools={len(self._unifai_tools) if self._unifai_tools else None}, tool_choice={self._unifai_tool_choice}, response_format={self._unifai_response_format})"


class Chat(BaseChat[ChatConfig]):
    config_class = ChatConfig
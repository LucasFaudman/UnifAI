from typing import Optional, Union, Sequence, Any, Literal, Mapping
from datetime import datetime

from ollama import Client as OllamaClient
from ollama._types import (
    ChatResponse as OllamaChatResponse,
    Message as OllamaMessage,
    Tool as OllamaTool, 
    ToolFunction as OllamaToolFunction,
    ToolCallFunction as OllamaToolCallFunction,
    ToolCall as OllamaToolCall, 
    Parameters as OllamaParameters, 
    Property as OllamaProperty,
    Options as OllamaOptions,
    RequestError as OllamaRequestError,
    ResponseError as OllamaResponseError,
)
from httpx import NetworkError, TimeoutException, HTTPError

from unifai._exceptions import (
    UnifAIError,
    APIError,
    UnknownAPIError,
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    STATUS_CODE_TO_EXCEPTION_MAP,   
)


from ._types import Message, Tool, ToolCall, Image, Usage, ResponseInfo
from ._convert_types import stringify_content
from .baseaiclientwrapper import BaseAIClientWrapper

from random import choices as random_choices
from string import ascii_letters, digits
from hashlib import sha256

def generate_random_id(length=8):
    return ''.join(random_choices(ascii_letters + digits, k=length))

def sha256_hash(string: str) -> str:
    return sha256(string.encode()).hexdigest()

class OllamaWrapper(BaseAIClientWrapper):
    client: OllamaClient

    def import_client(self):
        from ollama import Client as OllamaClient
        from ollama._types import Tool, ToolFunction, Parameters, Property, Message

        return OllamaClient
    
    def prep_input_message(self, message: Message) -> OllamaMessage:        
        role = message.role
        content = message.content or ''
        images = None # TODO: Implement image extraction
        tool_calls = None
        if message.images:
            images = [image.data for image in message.images if image.data]
        if message.tool_calls:
            if message.role == 'tool':
                tool_calls = None
            else:
                tool_calls = [self.prep_input_tool_call(tool_call) for tool_call in message.tool_calls]
            
        return OllamaMessage(role=role, content=content, images=images, tool_calls=tool_calls)

    def prep_input_messages_and_system_prompt(self, 
                                              messages: list[Message], 
                                              system_prompt_arg: Optional[str] = None
                                              ) -> tuple[list, Optional[str]]:
        if system_prompt_arg:
            system_prompt = system_prompt_arg
            if messages and messages[0].role == "system":
                messages[0].content = system_prompt
            else:
                messages.insert(0, Message(role="system", content=system_prompt))
        elif messages and messages[0].role == "system":
            system_prompt = messages[0].content
        else:
            system_prompt = None

        client_messages = [self.prep_input_message(message) for message in messages]
        return client_messages, system_prompt
            
    
    def prep_input_tool(self, tool: Tool) -> OllamaTool:
        # tool_dict = tool if isinstance(tool, dict) else tool.to_dict()

        tool_dict = tool.to_dict()
        tool_type = tool_dict['type']
        tool_def = tool_dict[tool_type]

        tool_name = tool_def['name']
        tool_description = tool_def['description']
        tool_parameters = tool_def['parameters']
        parameters_type = tool_parameters['type']
        parameters_required = tool_parameters['required']

        properties = {}
        for prop_name, prop_def in tool_parameters['properties'].items():
            prop_type = prop_def['type']
            prop_description = prop_def.get('description', None)
            prop_enum = prop_def.get('enum', None)
            properties[prop_name] = OllamaProperty(type=prop_type, description=prop_description, enum=prop_enum)
        
        parameters = OllamaParameters(type=parameters_type, required=parameters_required, properties=properties)
        tool_function = OllamaToolFunction(name=tool_name, description=tool_description, parameters=parameters)
        return OllamaTool(type=tool_type, function=tool_function)            
    
    def prep_input_tool_choice(self, tool_choice: str) -> str:
        return tool_choice
    
    def prep_input_tool_call(self, tool_call: ToolCall) -> OllamaToolCall:
        return OllamaToolCall(
            function=OllamaToolCallFunction(
                name=tool_call.tool_name, 
                arguments=tool_call.arguments if tool_call.arguments else {}
            )
        )
    
    
    def split_tool_call_outputs_into_messages(self, tool_calls: list[ToolCall], content: Optional[str] = None) -> list[Message]:        
        tool_messages = [
            Message(role="tool", content=stringify_content(tool_call.output), tool_calls=[tool_call])
            for tool_call in tool_calls
        ]
        if content:
            tool_messages.append(Message(role="user", content=content))
        return tool_messages

    # def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> OllamaMessage:
    #     return OllamaMessage(
    #         role='tool',
    #         content=tool_response,
    #     )

    def prep_input_response_format(self, response_format: Union[str, dict[str, str]]) -> str:
        if response_format in ("json", "json_object"):
            return 'json'
        return ''

    # def extract_output_tool_call(self, tool_call: OllamaToolCall) -> ToolCall:
    #     tool_call_function = tool_call['function']
    #     name = tool_call_function['name']
    #     arguments = tool_call_function.get('arguments')
    #     tool_call_id = f'call_{generate_random_id(24)}'
    #     return ToolCall(id=tool_call_id, tool_name=name, arguments=arguments)


    # def extract_output_message(self, response: OllamaChatResponse) -> Message:
    #     message = response['message']
    #     images = None # TODO: Implement image extraction
    #     tool_calls = [self.extract_output_tool_call(tool_call) for tool_call in message.get('tool_calls', [])]
    #     return Message(
    #         role=message['role'],
    #         content=message.get('content'),
    #         images=images,
    #         tool_calls=tool_calls,
    #         response_object=response
    #     )

    def extract_std_and_client_messages(self, response: OllamaChatResponse) -> tuple[Message, OllamaMessage]:
        client_message = response['message']
        
        # tool_calls = [self.extract_output_tool_call(tool_call) for tool_call in client_message.get('tool_calls', [])]

        tool_calls = None
        if client_message_tool_calls := client_message.get('tool_calls'):
            tool_calls = [
                ToolCall(
                    id=f'call_{generate_random_id(24)}',
                    tool_name=tool_call['function']['name'],
                    arguments=tool_call['function'].get('arguments')
                )
                for tool_call in client_message_tool_calls
            ]
        
        images = None # TODO: Implement image extraction

        model = response["model"]
        done_reason = response["done_reason"]
        if done_reason == "stop" and tool_calls:
            done_reason = "tool_calls"
        elif done_reason != "stop" and done_reason is not None:
            # TODO handle other done_reasons 
            done_reason = "max_tokens"
            

        # if choice.finish_reason == "length":
        #     done_reason = "max_tokens"
        # elif choice.finish_reason == "function_call":
        #     done_reason = "tool_calls"
        # else:
        #     # "stop", "tool_calls", "content_filter" or None
        #     done_reason = choice.finish_reason

        
        usage = Usage(
            input_tokens=response["prompt_eval_count"], 
            output_tokens=response["eval_count"]
        )

        created_at = datetime.strptime(f'{response["created_at"][:26]}Z', '%Y-%m-%dT%H:%M:%S.%fZ')
        response_info = ResponseInfo(model=model, done_reason=done_reason, usage=usage)          

        std_message = Message(
            role=client_message['role'],
            content=client_message.get('content'),            
            tool_calls=tool_calls,
            images=images,
            created_at=created_at,
            response_info=response_info
        )
        return std_message, client_message


    def convert_exception(self, exception: OllamaRequestError|OllamaResponseError|NetworkError|TimeoutError) -> UnifAIError:
        if isinstance(exception, OllamaRequestError):            
            status_code = 400
            message = exception.error
        elif isinstance(exception, OllamaResponseError):
            status_code = exception.status_code
            message = exception.error
        elif isinstance(exception, TimeoutException):
            status_code = 504
            message = exception.args[0]
        elif isinstance(exception, NetworkError):
            status_code = 502
            message = exception.args[0]            
        else:
            status_code = -1
            message = str(exception)
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message, 
            status_code=status_code,
            original_exception=exception
        )



    def list_models(self) -> Sequence[str]:
        return [model_name for model_dict in self.client.list()["models"] if (model_name := model_dict.get("name"))]
        
    def get_custom_model(self, base_model: str, system_prompt: str) -> str:
        model_name = f"{base_model}_{sha256_hash(system_prompt)}"
        if model_name not in self.list_models():
            modelfile = f'FROM {base_model}\nSYSTEM """{system_prompt}"""'
            prog_response = self.client.create(model=model_name, modelfile=modelfile)
            print(prog_response)
        return model_name
    
    def get_system_prompt_from_messages(self, messages: list[Message]) -> Optional[str]:
        if messages and messages[0].role == 'system':
            return messages[0].content


    def chat(
            self,
            messages: list[OllamaMessage],     
            model: Optional[str] = None, 
            system_prompt: Optional[str] = None,                   
            tools: Optional[list[OllamaTool]] = None,
            tool_choice: Optional[str] = None,
            response_format: Optional[str] = '',
            **kwargs
            ) -> OllamaChatResponse:
        
            # if (tool_choice and tool_choice != "auto" and system_prompt
            #     and messages and messages[0]["role"] == "system"
            #     ):                
            #     if tool_choice == "required":
            #         messages[0]["content"] = f"{system_prompt}\nYou MUST call one or more tools."
            #     elif tool_choice == "none":
            #         messages[0]["content"] = f"{system_prompt}\nYou CANNOT call any tools."
            #     else:
            #         messages[0]["content"] = f"{system_prompt}\nYou MUST call the tool '{tool_choice}' with ALL of its required arguments."
            #     system_prompt_modified = True
            # else:
            #     system_prompt_modified = False

            user_messages = [message for message in messages if message["role"] == 'user']
            last_user_content = user_messages[-1].get("content", "") if user_messages else ""            
            if (tool_choice and tool_choice != "auto" and last_user_content is not None
                ):                
                if tool_choice == "required":
                    user_messages[-1]["content"] = f"{last_user_content}\nYou MUST call one or more tools."
                elif tool_choice == "none":
                    user_messages[-1]["content"] = f"{last_user_content}\nYou CANNOT call any tools."
                else:
                    user_messages[-1]["content"] = f"{last_user_content}\nYou MUST call the tool '{tool_choice}' with ALL of its required arguments."
                last_user_content_modified = True
            else:
                last_user_content_modified = False

            keep_alive = kwargs.pop('keep_alive', None)
            stream = kwargs.pop('stream', False)
            options = kwargs.pop('options', None)
            if not options and kwargs:
                options = OllamaOptions(**kwargs)

            response = self.run_func_convert_exceptions(
                func=self.client.chat,
                messages=messages,                 
                model=model, 
                tools=tools, 
                format=response_format, 
                keep_alive=keep_alive,
                stream=stream,
                options=options
            )

            if last_user_content_modified:
                user_messages[-1]["content"] = last_user_content

            # if system_prompt_modified:
            #     response['message']['content'] = system_prompt

            # ollama-python is incorrectly typed as Mapping[str, Any] instead of OllamaChatResponse
            return response
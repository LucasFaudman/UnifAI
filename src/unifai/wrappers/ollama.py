from typing import Optional, Union, Sequence, Any, Literal, Mapping,  Iterator, Iterable, Generator, Collection
from datetime import datetime
from json import loads as json_loads, JSONDecodeError

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
from httpx import NetworkError, TimeoutException, HTTPError, RemoteProtocolError, TransportError

from unifai.exceptions import (
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


from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, Usage, ResponseInfo, Embeddings
from unifai.type_conversions import stringify_content
from ._base_llm_client import LLMClient
from ._base_embedding_client import EmbeddingClient

from random import choices as random_choices
from string import ascii_letters, digits
ASCII_LETTERS_AND_DIGITS = ascii_letters + digits

def generate_random_id(length: int = 8, chars: str = ASCII_LETTERS_AND_DIGITS):
    return ''.join(random_choices(chars, k=length))


class OllamaWrapper(EmbeddingClient, LLMClient):
    provider = "ollama"
    client: OllamaClient
    default_model = "mistral:7b-instruct"
    default_embedding_model = "llama3.1:8b-instruct-q2_K"

    def import_client(self):
        from ollama import Client as OllamaClient
        from ollama._types import Tool, ToolFunction, Parameters, Property, Message

        return OllamaClient


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: OllamaRequestError|OllamaResponseError|NetworkError|TimeoutError) -> UnifAIError:
        if isinstance(exception, OllamaRequestError):            
            message = exception.error
            status_code = 400
        elif isinstance(exception, OllamaResponseError):
            message = exception.error
            status_code = exception.status_code
        elif isinstance(exception, TimeoutException):
            message = exception.args[0]
            status_code = 504            
        elif isinstance(exception, TransportError):
            message = exception.args[0]
            status_code = 502 if '[Errno 65]' not in message else 504 # 502 is Bad Gateway, 504 is Gateway Timeout
        else:
            status_code = -1
            message = str(exception)
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message, 
            status_code=status_code,
            original_exception=exception
        )



    # List Models
    def list_models(self) -> Sequence[str]:
        return [model_name for model_dict in self.client.list()["models"] if (model_name := model_dict.get("name"))]
    

    # Embeddings
    def _get_embed_response(
            self,            
            input: str | Sequence[str],
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            **kwargs
            ) -> Mapping:
        
        model = model or self.default_embedding_model
        truncate = kwargs.pop('truncate', True)
        keep_alive = kwargs.pop('keep_alive', None)
        options = OllamaOptions(**kwargs) if kwargs else None
        return self.client.embed(
            input=input,
            model=model,
            truncate=truncate,
            keep_alive=keep_alive,
            options=options
        )


    def _extract_embeddings(
            self,            
            response: Any,
            model: str,
            max_dimensions: Optional[int] = None,
            **kwargs
            ) -> Embeddings:
        return Embeddings(
            root=[embedding[:max_dimensions] for embedding in response["embeddings"]],
            response_info=ResponseInfo(
                model=model, 
                usage=Usage(
                    input_tokens=response['prompt_eval_count'], 
                )
            )
        )     
    
        
    # Chat
    def _get_chat_response(
            self,
            stream: bool,            
            messages: list[OllamaMessage],     
            model: Optional[str] = None, 
            system_prompt: Optional[str] = None,                   
            tools: Optional[list[OllamaTool]] = None,
            tool_choice: Optional[str] = None,
            response_format: Optional[str] = '',
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,             
            **kwargs
            ) -> OllamaChatResponse|Iterator[OllamaChatResponse]:
        
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
                        
            if frequency_penalty is not None:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            if seed is not None:
                kwargs["seed"] = seed
            if stop_sequences is not None:
                kwargs["stop"] = stop_sequences
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_k is not None:
                kwargs["top_k"] = top_k
            if top_p is not None:
                kwargs["top_p"] = top_p

            options = OllamaOptions(**kwargs) if kwargs else None

            response = self.client.chat(
                model=model or self.default_model, 
                messages=messages,                 
                tools=tools, 
                stream=stream,
                format=response_format, 
                options=options,
                keep_alive=keep_alive,
            )

            if last_user_content_modified:
                user_messages[-1]["content"] = last_user_content

            # ollama-python is incorrectly typed as Mapping[str, Any] instead of OllamaChatResponse
            return response


    # Convert from UnifAI to AI Provider format        
        # Messages    
    def prep_input_user_message(self, message: Message) -> OllamaMessage:
        content = message.content or ''
        images = list(map(self.prep_input_image, message.images)) if message.images else None
        return OllamaMessage(role='user', content=content, images=images)


    def prep_input_assistant_message(self, message: Message) -> OllamaMessage:
        content = message.content or ''
        images = list(map(self.prep_input_image, message.images)) if message.images else None
        if message.tool_calls:
            tool_calls = [
                OllamaToolCall(
                    function=OllamaToolCallFunction(
                        name=tool_call.tool_name, 
                        arguments=tool_call.arguments
                    )
                ) 
                for tool_call in message.tool_calls
            ]
        else:
            tool_calls = None

        return OllamaMessage(role='assistant', content=content, images=images, tool_calls=tool_calls)
        

    def prep_input_tool_message(self, message: Message) -> OllamaMessage:
        if message.tool_calls:
            tool_call = message.tool_calls[0]        
            content = stringify_content(tool_call.output)
            images = list(map(self.prep_input_image, message.images)) if message.images else None
            tool_calls = None
            return OllamaMessage(role='tool', content=content, images=images, tool_calls=tool_calls) 
        
        raise ValueError("Tool message must have tool_calls") 


    def split_tool_message(self, message: Message) -> Iterator[Message]:        
        if tool_calls := message.tool_calls:
            for tool_call in tool_calls:
                yield Message(role="tool", tool_calls=[tool_call])
        if message.content is not None:
            yield Message(role="user", content=message.content)     


    def prep_input_system_message(self, message: Message) -> OllamaMessage:
        return OllamaMessage(role='system', content=message.content or '')
    

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
    
       
        # Images
    def prep_input_image(self, image: Image) -> Any:
        return image.raw_bytes


        # Tools
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


        # Response Format
    def prep_input_response_format(self, response_format: Literal["text", "json", "json_schema"]) -> Literal["", "json"]:
        if "json" in response_format:
            return "json"
        return ""


    # Convert Objects from AI Provider to UnifAI format    
        # Images
    def extract_image(self, response_image: Any, **kwargs) -> Image:
        raise NotImplementedError("This method must be implemented by the subclass")


        # Tool Calls
    def extract_tool_call(self, response_tool_call: OllamaToolCall, **kwargs) -> ToolCall:
        return ToolCall(
            id=f'call_{generate_random_id(24)}',
            tool_name=response_tool_call['function']['name'],
            arguments=response_tool_call['function'].get('arguments')
        )
    

        # Response Info (Model, Usage, Done Reason, etc.)
    def extract_done_reason(self, response_obj: OllamaChatResponse, **kwargs) -> str|None:        
        if not (done_reason := response_obj.get("done_reason")):
            return None
        elif done_reason == "stop" and response_obj["message"].get("tool_calls"):
            return "tool_calls"
        elif done_reason != "stop":
            # TODO handle other done_reasons 
            return "max_tokens"
        return done_reason
    

    def extract_usage(self, response_obj: OllamaChatResponse, **kwargs) -> Usage|None:
        return Usage(
            input_tokens=response_obj.get("prompt_eval_count", 0), 
            output_tokens=response_obj.get("eval_count", 0)
        )


    def extract_response_info(self, response: Any, **kwargs) -> ResponseInfo:
        model = response["model"]
        done_reason = self.extract_done_reason(response)
        usage = self.extract_usage(response)
        return ResponseInfo(model=model, done_reason=done_reason, usage=usage) 
    
    
        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def extract_assistant_message_both_formats(self, response: OllamaChatResponse, **kwargs) -> tuple[Message, OllamaMessage]:
        client_message = response['message']
        
        tool_calls = None
        if client_message_tool_calls := client_message.get('tool_calls'):
            tool_calls = list(map(self.extract_tool_call, client_message_tool_calls))

        
        images = None # TODO: Implement image extraction

        created_at = datetime.strptime(f'{response["created_at"][:26]}Z', '%Y-%m-%dT%H:%M:%S.%fZ')
        response_info = self.extract_response_info(response)        

        std_message = Message(
            role=client_message['role'],
            content=client_message.get('content'),            
            tool_calls=tool_calls,
            images=images,
            created_at=created_at,
            response_info=response_info
        )
        return std_message, client_message
    

    def extract_stream_chunks(self, response: Iterator[OllamaChatResponse], **kwargs) -> Generator[MessageChunk, None, tuple[Message, OllamaMessage]]:
        content = ""
        tool_calls = []
        model = None
        usage = None
        done_reason = None
        lbrackets, rbrackets = 0, 0
        for chunk in response:
            if not model:
                model = chunk.get("model")
            if not done_reason:
                done_reason = self.extract_done_reason(chunk)
            if not usage:
                usage = self.extract_usage(chunk)

            if (chunk_message := chunk.get("message")) and (chunk_content := chunk_message.get("content")):
                    content += chunk_content
                    lbrackets += chunk_content.count("{")
                    rbrackets += chunk_content.count("}")
                    if lbrackets and lbrackets == rbrackets:
                        try:
                            tool_call_dict = json_loads(content)
                            tool_call = ToolCall(
                                id=f'call_{generate_random_id(24)}',
                                tool_name=tool_call_dict['name'],
                                arguments=tool_call_dict.get('parameters')
                            )
                            tool_calls.append(tool_call)
                            yield MessageChunk(
                                role="assistant", 
                                tool_calls=[tool_call]
                            )
                            content = ""
                            lbrackets, rbrackets = 0, 0
                        except JSONDecodeError as e:
                            yield MessageChunk(
                                role="assistant", 
                                content=chunk_content
                            )
                    elif lbrackets > rbrackets:
                        continue
                    else:
                        yield MessageChunk(
                            role="assistant", 
                            content=chunk_content
                        )
                
        response_info = ResponseInfo(model=model, done_reason=done_reason, usage=usage)
        std_message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            response_info=response_info
        )
        return std_message, self.prep_input_assistant_message(std_message)




        

        
   


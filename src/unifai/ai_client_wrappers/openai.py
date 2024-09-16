# from openai import OpenAI, OpenAIError, BadRequestError
# from openai.types.beta import Assistant, Thread
# from openai.types.beta.threads import Message, Run, Text
# from openai.types.beta.threads.run import LastError
# from openai.types.beta.threads.runs import RunStep
# from openai.types.beta.threads.runs.message_creation_step_details import (
#     MessageCreationStepDetails,
# )
# from openai.types.beta.threads.runs.tool_calls_step_details import ToolCallsStepDetails
# from openai.types.beta.threads.file_citation_annotation import FileCitationAnnotation
# from openai.types.beta.threads.text_content_block import TextContentBlock
# from openai.types.beta.threads.image_file_content_block import ImageFileContentBlock
# from openai.types.beta.threads.file_path_annotation import FilePathAnnotation
# from openai.types.beta.threads.runs.function_tool_call import FunctionToolCall, Function
# from openai.types.beta.threads.runs.code_interpreter_tool_call import (
#     CodeInterpreterToolCall,
# )
# from openai.types.beta.code_interpreter_tool import CodeInterpreterTool
# from openai.types.beta.function_tool import FunctionTool

# from tiktoken import get_encoding, encoding_for_model
from typing import Optional, Union, Any, Literal, Mapping, Iterator
from json import loads as json_loads, JSONDecodeError
from datetime import datetime


from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai import (
    OpenAIError,
    APIError as OpenAIAPIError,
    APIConnectionError as OpenAIAPIConnectionError,
    APITimeoutError as OpenAIAPITimeoutError,
    APIResponseValidationError as OpenAIAPIResponseValidationError,
    APIStatusError as OpenAIAPIStatusError,
    AuthenticationError as OpenAIAuthenticationError,
    BadRequestError as OpenAIBadRequestError,
    ConflictError as OpenAIConflictError,
    InternalServerError as OpenAIInternalServerError,
    NotFoundError as OpenAINotFoundError,
    PermissionDeniedError as OpenAIPermissionDeniedError,
    RateLimitError as OpenAIRateLimitError,
    UnprocessableEntityError as OpenAIUnprocessableEntityError,
)


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
    STATUS_CODE_TO_EXCEPTION_MAP   
)

from unifai.types import Message, Tool, ToolCall, Image, Usage, ResponseInfo
from unifai.type_conversions import stringify_content
from ._base import BaseAIClientWrapper

class OpenAIWrapper(BaseAIClientWrapper):
    client: OpenAI
    default_model = "gpt-4o-mini"

    def import_client(self):
        from openai import OpenAI
        return OpenAI
    
    
    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: OpenAIAPIError) -> UnifAIError:
        if isinstance(exception, OpenAIAPIResponseValidationError):
            return APIResponseValidationError(
                message=exception.message,
                status_code=exception.status_code, # Status code could be anything
                error_code=exception.code,
                original_exception=exception,
            )
        
        if isinstance(exception, OpenAIAPITimeoutError):
            status_code = 504
        elif isinstance(exception, OpenAIAPIConnectionError):
            status_code = 502
        else:
            status_code = getattr(exception, "status_code", -1)
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=exception.message, 
            status_code=getattr(exception, "status_code", None), # ConnectionError and TimeoutError don't have status_code
            error_code=exception.code, 
            original_exception=exception
        )
        
    
    # Convert from UnifAI to AI Provider format        
        # Messages        
    def prep_input_user_message(self, message: Message) -> dict:
        message_dict = {"role": "user", "content": message.content}
        if message.images:
            message_dict["images"] = self.prep_input_images(message.images)
        return message_dict
    

    def prep_input_assistant_message(self, message: Message) -> dict:
        message_dict = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    tool_call.type: {
                        "name": tool_call.tool_name,
                        "arguments": stringify_content(tool_call.arguments),
                    }
                }
                for tool_call in message.tool_calls
            ]
        if message.images:
            message_dict["images"] = self.prep_input_images(message.images)
        
        return message_dict   
    

    def prep_input_tool_message(self, message: Message) -> dict:
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": message.content,
            }
        raise ValueError("Tool message must have tool_calls")
    
    
    def prep_input_system_message(self, message: Message) -> dict:
        return {"role": "system", "content": message.content}


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
    def prep_input_images(self, images: list[Image]) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")    
    
    
        # Tools
    def prep_input_tool(self, tool: Tool) -> dict:
        return tool.to_dict()
    
    
    def prep_input_tool_choice(self, tool_choice: str) -> Union[str, dict]:
        if tool_choice in ("auto", "required", "none"):
            return tool_choice

        tool_type = "function" # Currently only function tools are supported See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        return {"type": tool_type, tool_type: {"name": tool_choice}}


        # Response Format
    def prep_input_response_format(self, response_format: Union[str, dict]) -> Union[str, dict]:

        if isinstance(response_format, dict) and (response_type := response_format.get("type")):
            if response_type == "json_schema" and (schema := response_format.get("json_schema")):
                # TODO handle json_schema
                # schema = handle_json_schema(schema)
                return {"type": response_type, response_type: schema}
        else:
            response_type = response_format    
        
        if response_type in ("json", "json_object", "text"):
            return {"type": response_type}
        
        raise ValueError(f"Invalid response_format: {response_format}")
        
    
    # Convert from AI Provider to UnifAI format   
    def extract_output_assistant_messages(self, response: ChatCompletion) -> tuple[Message, ChatCompletionMessage]:
        choice = response.choices[0]
        client_message = choice.message

        tool_calls = None
        if client_message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=json_loads(tool_call.function.arguments),
                )
                for tool_call in client_message.tool_calls
            ]

        images = None # TODO: Implement image extraction

        model = response.model
        if choice.finish_reason == "length":
            done_reason = "max_tokens"
        elif choice.finish_reason == "function_call":
            done_reason = "tool_calls"
        else:
            # "stop", "tool_calls", "content_filter" or None
            done_reason = choice.finish_reason

        if response.usage:
            usage = Usage(input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)
        else:
            usage = None
        
        created_at = datetime.fromtimestamp(response.created)
        response_info = ResponseInfo(model=model, done_reason=done_reason, usage=usage)        
        
        std_message = Message(
            role=client_message.role,
            content=client_message.content,            
            tool_calls=tool_calls,
            images=images,
            created_at=created_at,
            response_info=response_info,
        )   
        return std_message, client_message


    def split_tool_outputs_into_messages(self, tool_calls: list[ToolCall], content: Optional[str] = None) -> Iterator[Message]:        
        for tool_call in tool_calls:
            yield Message(role="tool", content=stringify_content(tool_call.output), tool_calls=[tool_call])        
        if content is not None:
            yield Message(role="user", content=content)





    # List Models
    def list_models(self) -> list[str]:
        return [model.id for model in self.client.models.list()]

    # Chat 
    def chat(
            self,
            messages: list[dict],     
            model: str = default_model,
            system_prompt: Optional[str] = None,                    
            tools: Optional[list[dict]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            response_format: Optional[str] = None,

            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None, 

            **kwargs
            ) -> ChatCompletion:

            if tool_choice and not tools:
                tool_choice = None

            response = self.run_func_convert_exceptions(
                func=self.client.chat.completions.create,
                messages=messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,

                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop_sequences,
                temperature=temperature,
                # top_k=top_k,
                top_p=top_p,
                **kwargs
            )            
            return response


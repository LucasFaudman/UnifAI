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
from typing import Optional, Union, Any, Literal, Mapping, Iterator, Sequence, Generator
from json import loads as json_loads, JSONDecodeError
from datetime import datetime


from openai import OpenAI
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDeltaToolCall, Choice as ChunkChoice
from openai.types.chat.chat_completion import ChatCompletion, Choice as CompletionChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.create_embedding_response import CreateEmbeddingResponse
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
    STATUS_CODE_TO_EXCEPTION_MAP,
)

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, Usage, ResponseInfo, Embeddings, EmbeddingTaskTypeInput
from unifai.type_conversions import stringify_content
from ._base_llm_client import LLMClient
from ._base_embedder import Embedder

class OpenAIWrapper(Embedder, LLMClient):
    provider = "openai"
    client: OpenAI
    default_model = "gpt-4o"
    default_embedding_model = "text-embedding-3-small"

    model_embedding_dimensions = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1568,
    }    

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
        
        message = getattr(exception, "message", str(exception))
        error_code = getattr(exception, "code", None)
        if isinstance(exception, OpenAIAPITimeoutError):
            status_code = 504            
        elif isinstance(exception, OpenAIAPIConnectionError):                
            status_code = 502
        elif isinstance(exception, OpenAIAPIStatusError):
            status_code = getattr(exception, "status_code", -1)
        else:
            status_code = 401 if "api_key" in message else getattr(exception, "status_code", -1)
        #TODO model does not support tool calls, images, etc feature errors

        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message, 
            status_code=status_code,
            error_code=error_code, 
            original_exception=exception
        )
        


    # List Models
    def list_models(self) -> list[str]:
        return [model.id for model in self.client.models.list()]


    # Embeddings
    def _get_embed_response(
            self,
            input: Sequence[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Literal["search_query", "search_document", "classification", "clustering", "image"] = "search_query",
            input_too_large: Literal[
                "truncate_end", 
                "truncate_start", 
                "raise_error"] = "truncate_end",
            **kwargs
            ) -> CreateEmbeddingResponse:
        
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        return self.client.embeddings.create(input=input, model=model, **kwargs)


    def _extract_embeddings(
            self,            
            response: CreateEmbeddingResponse,
            **kwargs
            ) -> Embeddings:
        return Embeddings(
            root=[embedding.embedding for embedding in response.data],
            response_info=ResponseInfo(
                model=response.model, 
                usage=Usage(
                    input_tokens=response.usage.prompt_tokens, 
                    output_tokens=0
                )
            )
        )

   
    def _create_completion(self, kwargs) -> ChatCompletion|Stream[ChatCompletionChunk]:
        return self.client.chat.completions.create(**kwargs)

                
    # Chat 
    def _get_chat_response(
            self,
            stream: bool,            
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
            ) -> ChatCompletion|Stream[ChatCompletionChunk]:

            if stream:
                kwargs["stream"] = stream
                kwargs["stream_options"] = kwargs.get("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True
            
            kwargs["messages"] = messages
            kwargs["model"] = model
            if tools:
                kwargs["tools"] = tools
            if tool_choice and tools:
                kwargs["tool_choice"] = tool_choice
            if response_format:
                kwargs["response_format"] = response_format
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            if frequency_penalty:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty:
                kwargs["presence_penalty"] = presence_penalty
            if seed:
                kwargs["seed"] = seed
            if stop_sequences:
                kwargs["stop"] = stop_sequences
            if temperature:
                kwargs["temperature"] = temperature
            # if top_k:
            #     kwargs["top_k"] = top_k
            if top_p:
                kwargs["top_p"] = top_p
            
            return self._create_completion(kwargs)  
    
       
    
    # Convert from UnifAI to AI Provider format        
        # Messages        
    def prep_input_user_message(self, message: Message) -> dict:
        message_dict = {"role": "user"}        
        if message.images:
            content = []
            if message.content:
                content.append({"type": "text", "text": message.content})                
            content.extend(map(self.prep_input_image, message.images))
        else:
            content = message.content
        
        message_dict["content"] = content
        return message_dict
    

    def prep_input_assistant_message(self, message: Message) -> dict:
        message_dict = {"role": "assistant", "content": message.content or ""}
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
            message_dict["images"] = list(map(self.prep_input_image, message.images))
        
        return message_dict   
    

    def prep_input_tool_message(self, message: Message) -> dict:
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": stringify_content(tool_call.output),
            }
        raise ValueError("Tool message must have tool_calls")
    
    
    def split_tool_message(self, message: Message) -> Iterator[Message]:        
        if tool_calls := message.tool_calls:
            for tool_call in tool_calls:
                yield Message(role="tool", tool_calls=[tool_call])
        if message.content is not None:
            yield Message(role="user", content=message.content) 

    
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

        client_messages = []
        for message in messages:
            if message.role != "tool":
                client_messages.append(self.prep_input_message(message))
            else:
                client_messages.extend(map(self.prep_input_message, self.split_tool_message(message)))

        return client_messages, system_prompt


        # Images
    def prep_input_image(self, image: Image) -> dict:
        if not (image_url := image.url):
            image_url = image.data_uri         
        return {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}


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
        
    
    # Convert Objects from AI Provider to UnifAI format    
        # Images
    def extract_image(self, response_image: Any, **kwargs) -> Image:
        raise NotImplementedError("This method must be implemented by the subclass")

        # Tool Calls
    def extract_tool_call(self, response_tool_call: ChatCompletionMessageToolCall|ChoiceDeltaToolCall) -> ToolCall:
        return ToolCall(
            id=response_tool_call.id,
            tool_name=response_tool_call.function.name,
            arguments=json_loads(response_tool_call.function.arguments),
        )
    

    def extract_done_reason(self, response_obj: CompletionChoice|ChunkChoice, **kwargs) -> str|None:
        done_reason = response_obj.finish_reason
        if done_reason == "length":
            return "max_tokens"
        if done_reason == "function_call":
            return "tool_calls"
        
        # "stop", "tool_calls", "content_filter" or None
        return done_reason
    
    
    def extract_usage(self, response_obj: Any, **kwargs) -> Usage|None:
        if response_usage := response_obj.usage:
            return Usage(input_tokens=response_usage.prompt_tokens, output_tokens=response_usage.completion_tokens)


        # Response Info (Model, Usage, Done Reason, etc.)
    def extract_response_info(self, response: Any, **kwargs) -> ResponseInfo:
        model = response.model or kwargs.get("model")
        done_reason = self.extract_done_reason(response.choices[0])
        usage = self.extract_usage(response)
        
        return ResponseInfo(model=model, done_reason=done_reason, usage=usage) 
    
    
        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def extract_assistant_message_both_formats(self, response: ChatCompletion, **kwargs) -> tuple[Message, ChatCompletionMessage]:
        client_message = response.choices[0].message

        tool_calls = None
        if client_message.tool_calls:
            tool_calls = list(map(self.extract_tool_call, client_message.tool_calls))

        # if client_message.images:
        #     images = self.extract_images(client_message.images)
        images = None

        created_at = datetime.fromtimestamp(response.created) if response.created else datetime.now()
        response_info = self.extract_response_info(response, **kwargs)       
        
        std_message = Message(
            role=client_message.role,
            content=client_message.content,            
            tool_calls=tool_calls,
            images=images,
            created_at=created_at,
            response_info=response_info,
        )   
        return std_message, client_message


    def extract_stream_chunks(self, response: Stream[ChatCompletionChunk], **kwargs) -> Generator[MessageChunk, None, tuple[Message, ChatCompletionMessage]]:        
        content = ""
        tool_calls = []
        last_tool_call_yielded = -1
        model = None
        usage = None
        done_reason = None
        
        for chunk in response:
            if chunk.usage:
                usage = self.extract_usage(chunk)
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            if not model:
                model = chunk.model
            if choice.finish_reason:
                done_reason = self.extract_done_reason(choice)
            if delta_content := delta.content:
                content += delta_content    
                yield MessageChunk(
                    role="assistant", 
                    content=delta_content, 
                    response_info=ResponseInfo(model=chunk.model, done_reason=done_reason), 
                    created_at=datetime.fromtimestamp(chunk.created)
                )                               
            
            if delta_tool_calls := delta.tool_calls:
                tool_call_delta = delta_tool_calls[0]
                index = tool_call_delta.index
                len_tool_calls = len(tool_calls)
                if index == len_tool_calls:
                    # if tool_calls:
                    #     tool_calls[-1] = self.extract_tool_call(tool_calls[-1])
                    #     yield MessageChunk(
                    #         role="assistant", 
                    #         tool_calls=[tool_calls[-1]], 
                    #         response_info=ResponseInfo(model=chunk.model, done_reason=done_reason), 
                    #         created_at=datetime.fromtimestamp(chunk.created)
                    #     )
                    #     last_tool_call_yielded = index - 1                        
                    tool_calls.append(tool_call_delta)
                elif index < len_tool_calls:
                     tool_calls[index].function.arguments += tool_call_delta.function.arguments
                

        for i, tool_call in enumerate(tool_calls[last_tool_call_yielded + 1:], start=last_tool_call_yielded + 1):
            tool_call = self.extract_tool_call(tool_call)
            tool_calls[i] = tool_call
            yield MessageChunk(
                role="assistant", 
                tool_calls=[tool_call], 
                response_info=ResponseInfo(model=chunk.model, done_reason=done_reason), 
                created_at=datetime.fromtimestamp(chunk.created)
            )
                    
                
        response_info = ResponseInfo(model=model, done_reason=done_reason, usage=usage)
        std_message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            response_info=response_info
        )
        return std_message, self.prep_input_message(std_message)







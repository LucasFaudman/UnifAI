from typing import Optional, Union, Sequence, Any, Literal, Mapping, Iterator

from anthropic import Anthropic
from anthropic.types import (
    Usage as AnthropicUsage,    
    Message as AnthropicMessage,
    MessageParam as AnthropicMessageParam,
    ContentBlock as AnthropicContentBlock,
    TextBlock as AnthropicTextBlock,
    ToolUseBlock as AnthropicToolUseBlock,
    TextBlockParam as AnthropicTextBlockParam,
    ImageBlockParam as AnthropicImageBlockParam,
    ToolUseBlockParam as AnthropicToolUseBlockParam,
    ToolResultBlockParam as AnthropicToolResultBlockParam,
    ToolParam as AnthropicToolParam, 
)
from anthropic.types.image_block_param import Source as AnthropicImageSource


from anthropic import (
    AnthropicError,
    APIError as AnthropicAPIError,
    APIConnectionError as AnthropicAPIConnectionError,
    APITimeoutError as AnthropicAPITimeoutError,
    APIResponseValidationError as AnthropicAPIResponseValidationError,
    APIStatusError as AnthropicAPIStatusError,
    AuthenticationError as AnthropicAuthenticationError,
    BadRequestError as AnthropicBadRequestError,
    ConflictError as AnthropicConflictError,
    InternalServerError as AnthropicInternalServerError,
    NotFoundError as AnthropicNotFoundError,
    PermissionDeniedError as AnthropicPermissionDeniedError,
    RateLimitError as AnthropicRateLimitError,
    UnprocessableEntityError as AnthropicUnprocessableEntityError,
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

class AnthropicWrapper(BaseAIClientWrapper):
    # client: Anthropic
    default_model = "claude-3-5-sonnet-20240620"

    def import_client(self):
        from anthropic import Anthropic
        return Anthropic


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: AnthropicAPIError) -> UnifAIError:
        if isinstance(exception, AnthropicAPIResponseValidationError):
            return APIResponseValidationError(
                message=exception.message,
                status_code=exception.status_code, # Status code could be anything
                original_exception=exception,
            )
        
        if isinstance(exception, AnthropicAPIError):
            message = exception.message
            if isinstance(exception, AnthropicAPITimeoutError):
                status_code = 504
            elif isinstance(exception, AnthropicAPIConnectionError):
                status_code = 502
            else:
                status_code = getattr(exception, "status_code", -1)
        else:
            message = str(exception)
            status_code = 401 if "api_key" in message else -1
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message,
            status_code=status_code,
            original_exception=exception
        )
    

    # Convert Objects from UnifAI to AI Provider format            
        # Messages
    def prep_input_user_message(self, message: Message) -> AnthropicMessageParam:
        content = []
        if message.images:
            content.extend(map(self.prep_input_image, message.images))        
        if message.content:
            content.append(AnthropicTextBlockParam(text=message.content, type="text"))

        return AnthropicMessageParam(role="user", content=content)


    def prep_input_assistant_message(self, message: Message) -> AnthropicMessageParam:
        content = []
        if message.content:
            content.append(AnthropicTextBlockParam(text=message.content, type="text"))
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_use_param = AnthropicToolUseBlockParam(
                    id=tool_call.id,
                    name=tool_call.tool_name,
                    input=tool_call.arguments,
                    type="tool_use"
                )
                content.append(tool_use_param)
        if message.images:
            content.extend(map(self.prep_input_image, message.images))
        return AnthropicMessageParam(role="assistant", content=content)


    def prep_input_tool_message(self, message: Message) -> AnthropicMessageParam:
        content = []
        # if message.content:
        #     content.append(AnthropicTextBlockParam(text=message.content, type="text"))
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_result_content = []
                # if message.content:
                #     tool_result_content.append(AnthropicTextBlockParam(text=message.content, type="text"))                

                tool_result_param = AnthropicToolResultBlockParam(
                    tool_use_id=tool_call.id,
                    content=tool_result_content,
                    is_error=False,
                    type="tool_result"
                )
                content.append(tool_result_param)
            return AnthropicMessageParam(role="user", content=content)
        raise ValueError("Tool messages must have tool calls")


    def prep_input_system_message(self, message: Message) -> AnthropicMessageParam:
        raise ValueError("Anthropic does not support system messages")
    

    def prep_input_messages_and_system_prompt(self, 
                                              messages: list[Message], 
                                              system_prompt_arg: Optional[str] = None
                                              ) -> tuple[list, Optional[str]]:
        system_prompt = system_prompt_arg
        if messages and messages[0].role == "system":
            # Remove the first system message from the list since Anthropic does not support system messages
            system_message = messages.pop(0)
            # Set the system prompt to the content of the first system message if not set by the argument
            if not system_prompt:
                system_prompt = system_message.content 

        client_messages = [self.prep_input_message(message) for message in messages]
        return client_messages, system_prompt        


        # Images
    def prep_input_image(self, image: Image) -> AnthropicImageBlockParam:
        return AnthropicImageBlockParam(
            type="image",
            source=AnthropicImageSource(
                type="base64",
                data=image.path or image.base64_string,
                media_type=image.media_type                
            )
        )


        # Tools
    def prep_input_tool(self, tool: Tool) -> AnthropicToolParam:
        return AnthropicToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters.to_dict()
        )    


    def prep_input_tool_choice(self, tool_choice: str) -> dict:
        if tool_choice == "any":
            tool_choice = "required"
        if tool_choice in ("auto", "required", "none"):
            return {"type": tool_choice}
        
        return {"type": "tool", "name": tool_choice}   


        # Response Format
    def prep_input_response_format(self, response_format: Union[str, dict]) -> None:
        # Warn: response_format is not used by the Anthropic client
        if response_format: print("Warning: response_format is not used by the Anthropic client")
        return None


    # Convert Objects from AI Provider to UnifAI format    
        # Images
    def extract_image(self, response_image: Any) -> Image:
        raise NotImplementedError("This method must be implemented by the subclass")

        # Tool Calls
    def extract_tool_call(self, response_tool_call: AnthropicToolUseBlock) -> ToolCall:
        return ToolCall(
                    id=response_tool_call.id,
                    tool_name=response_tool_call.name,
                    arguments=response_tool_call.input 
                )
    
        # Response Info (Model, Usage, Done Reason, etc.)
    def extract_response_info(self, response: AnthropicMessage) -> ResponseInfo:
        model = response.model
        if response.stop_reason == "end_turn" or response.stop_reason == "stop_sequence":
            done_reason = "stop"
        elif response.stop_reason == "tool_use":
            done_reason = "tool_calls"
        else:
            # "max_tokens" or None
            done_reason = response.stop_reason

        if response.usage:
            usage = Usage(
                input_tokens=response.usage.input_tokens, 
                output_tokens=response.usage.output_tokens
            )
        else:
            usage = None                      

        return ResponseInfo(model=model, done_reason=done_reason, usage=usage) 


    # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def extract_assistant_message_both_formats(self, response: AnthropicMessage) -> tuple[Message, AnthropicMessageParam]:
        client_message = AnthropicMessageParam(role=response.role, content=response.content)
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(self.extract_tool_call(block))

        images = None # TODO: Implement image extraction  
        response_info = self.extract_response_info(response)

        std_message = Message(
            role=response.role,
            content=content or None,
            tool_calls=tool_calls or None,
            images=images or None,
            response_info=response_info,
        )
        return std_message, client_message 
    

    def split_tool_outputs_into_messages(self, 
                                              tool_calls: list[ToolCall],
                                              content: Optional[str],
                                              ) -> Iterator[Message]:        
        yield Message(
            role="tool",
            content=content,
            tool_calls=tool_calls            
        )
        


    # List Models
    def list_models(self) -> list[str]:
        claude_models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
        return claude_models
        


    # Chat 
    def chat(
            self,
            messages: list[AnthropicMessageParam],     
            model: str = default_model, 
            system_prompt: Optional[str] = None,                 
            tools: Optional[list[AnthropicToolParam]] = None,
            tool_choice: Optional[dict] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,

            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,             
            **kwargs
            ) -> tuple[Message, AnthropicMessageParam]:
        
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        max_tokens = max_tokens or 4096
        if stop_sequences is not None:
            kwargs["stop_sequences"] = stop_sequences
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_k is not None:
            kwargs["top_k"] = top_k
        if top_p is not None:
            kwargs["top_p"] = top_p

        response = self.run_func_convert_exceptions(
            func=self.client.messages.create,
            max_tokens=max_tokens,
            messages=messages,
            model=model,

            **kwargs
        )
        return self.extract_assistant_message_both_formats(response)

    # Embeddings
    def embeddings(
            self,
            model: Optional[str] = None,
            texts: Optional[Sequence[str]] = None,
            **kwargs
            ):
        raise NotImplementedError("This method must be implemented by the subclass")
    
    # Generate

    def generate(
            self,
            model: Optional[str] = None,
            prompt: Optional[str] = None,
            **kwargs
            ):
        raise NotImplementedError("This method must be implemented by the subclass")


    def create_assistant(self, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")
        
    def update_assistant(self, ass_id, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def create_thread(self):
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def create_run(self):
        raise NotImplementedError("This method must be implemented by the subclass")    
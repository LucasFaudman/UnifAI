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
 # Anthropic
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
    STATUS_CODE_TO_EXCEPTION_MAP   
)


from ._types import Message, Tool, ToolCall, Image, Usage, ResponseInfo
from ._convert_types import stringify_content
from .baseaiclientwrapper import BaseAIClientWrapper

class AnthropicWrapper(BaseAIClientWrapper):
    client: Anthropic
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
        
        if isinstance(exception, AnthropicAPITimeoutError):
            status_code = 504
        elif isinstance(exception, AnthropicAPIConnectionError):
            status_code = 502
        else:
            status_code = getattr(exception, "status_code", -1)
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=exception.message, 
            status_code=getattr(exception, "status_code", None), # ConnectionError and TimeoutError don't have status_code
            original_exception=exception
        )
    

    # Convert Objects from UnifAI to AI Provider format            
        # Messages
    def prep_input_user_message(self, message: Message) -> AnthropicMessageParam:
        content = []
        if message.content:
            content.append(AnthropicTextBlockParam(text=message.content, type="text"))
        if message.images:
            content.extend(self.prep_input_images(message.images))

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
        # if message.images:
        #     content.extend(self.prep_input_images(message.images))
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
    def prep_input_images(self, images: list[Image]) -> list[AnthropicImageBlockParam]:
        raise NotImplementedError("This method must be implemented by the subclass")    


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


    # Convert from AI Provider to UnifAI format    
    def extract_output_assistant_messages(self, response: AnthropicMessage) -> tuple[Message, AnthropicMessageParam]:
        client_message = AnthropicMessageParam(role=response.role, content=response.content)
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    tool_name=block.name,
                    arguments=block.input 
                ))

        images = None # TODO: Implement image extraction  

        model = response.model
        if response.stop_reason == "end_turn" or response.stop_reason == "stop_sequence":
            done_reason = "stop"
        elif response.stop_reason == "tool_use":
            done_reason = "tool_calls"
        else:
            # "max_tokens" or None
            done_reason = response.stop_reason

        if response.usage:
            usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        else:
            usage = None                      

        response_info = ResponseInfo(model=model, done_reason=done_reason, usage=usage)       
        std_message = Message(
            role=response.role,
            content=content,
            tool_calls=tool_calls,
            images=images,
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
            **kwargs
            ) -> AnthropicMessage:
        
        max_tokens = kwargs.pop("max_tokens", 4096)
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        response = self.run_func_convert_exceptions(
            func=self.client.messages.create,
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            **kwargs
        )
        return response

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
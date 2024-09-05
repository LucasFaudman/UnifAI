from typing import Optional, Union, Sequence, Any, Literal, Mapping

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



from ._types import Message, Tool, ToolCall, Image
from ._convert_types import stringify_content
from .baseaiclientwrapper import BaseAIClientWrapper
from io import BytesIO

class AnthropicWrapper(BaseAIClientWrapper):
    client: Anthropic
    default_model = "claude-3-5-sonnet-20240620"

    def import_client(self):
        from anthropic import Anthropic
        return Anthropic


    def prep_input_message(self, message: Message) -> AnthropicMessageParam:
        if message.role == "system":
            raise ValueError("Anthropic does not support system messages")
        
        role = "user" if message.role != "assistant" else message.role
        content = []


        if message.tool_calls:
            if message.role == "tool":
                content = self.prep_input_message_tool_call_response(message, content)
            else:
                content = self.prep_input_message_tool_call(message, content)

        if message.content and not content:
            content.append(AnthropicTextBlockParam(text=message.content, type="text"))            

        if message.images and not message.tool_calls:
            # TODO: Implement image handling
            for image in message.images:
                source = AnthropicImageSource(data=image.data, media_type=image.media_type, type=image.format), 
                content.append(AnthropicImageBlockParam(source=source, type="image"))

        return AnthropicMessageParam(role=role, content=content)     
            

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
    
    def prep_input_tool(self, tool: Tool) -> AnthropicToolParam:
        return AnthropicToolParam(
            name=tool.name,
            description=tool.description,
            input_schema=tool.parameters.to_dict()
        )
        
    # def prep_input_tool_call(self, tool_call: ToolCall) -> AnthropicToolUseBlockParam:
    #     return AnthropicToolUseBlockParam(
    #         id=tool_call.tool_call_id,
    #         name=tool_call.tool_name,
    #         input=tool_call.arguments,
    #         type="tool_use"
    #     )

    # def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> AnthropicToolResultBlockParam:
    #     raise NotImplementedError("This method must be implemented by the subclass")


    def prep_input_message_tool_call(self, message: Message, input_object: list) -> list:
        if message.tool_calls and message.role == "assistant":
            for tool_call in message.tool_calls:
                tool_use_param = AnthropicToolUseBlockParam(
                    id=tool_call.tool_call_id,
                    name=tool_call.tool_name,
                    input=tool_call.arguments,
                    type="tool_use"
                )
                input_object.append(tool_use_param)                
        return input_object

    
    def split_tool_call_outputs_into_messages(self, tool_calls: list[ToolCall]) -> list[Message]:        
        message = Message(
            role="tool",
            tool_calls=tool_calls,            
        )
        return [message]
    
    def prep_input_message_tool_call_response(self, message: Message, input_object: list) -> list:
        if message.tool_calls and message.role == "tool":
            for tool_call in message.tool_calls:
                content = []
                if message.content:
                    content.append(AnthropicTextBlockParam(text=message.content, type="text"))
                if message.images:
                    for image in message.images:
                        source = AnthropicImageSource(data=image.data, media_type=image.media_type, type=image.format), 
                        content.append(AnthropicImageBlockParam(source=source, type="image"))
                
                tool_result_param = AnthropicToolResultBlockParam(
                    tool_use_id=tool_call.tool_call_id,
                    content=content,
                    is_error=False,
                    type="tool_result"
                )
                input_object.append(tool_result_param)
        return input_object


    def prep_input_tool_choice(self, tool_choice: str) -> dict:
        if tool_choice == "any":
            tool_choice = "required"
        if tool_choice in ("auto", "required", "none"):
            return {"type": tool_choice}
        
        return {"type": "tool", "name": tool_choice}

    def prep_input_response_format(self, response_format: Union[str, dict]) -> None:
        # Warn: response_format is not used by the Anthropic client
        if response_format: print("Warning: response_format is not used by the Anthropic client")
        return None

            
    def extract_output_tool_call(self, tool_call: AnthropicToolUseBlock) -> ToolCall:
        return ToolCall(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=tool_call.input
        )
    
    def extract_std_and_client_messages(self, response: AnthropicMessage) -> tuple[Message, AnthropicMessageParam]:
        client_message = AnthropicMessageParam(role=response.role, content=response.content)
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(self.extract_output_tool_call(block))

        std_message = Message(
            role=response.role,
            content=content,
            tool_calls=tool_calls,
            response_object=response
        )

        return std_message, client_message

        
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

        response = self.client.messages.create(
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
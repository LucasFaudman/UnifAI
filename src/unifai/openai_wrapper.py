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
from typing import Optional, Union, Any, Literal, Mapping
from json import loads as json_loads, JSONDecodeError



from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from ._types import Message, Tool, ToolCall
from ._convert_types import stringify_content
from .baseaiclientwrapper import BaseAIClientWrapper

class OpenAIWrapper(BaseAIClientWrapper):
    client: OpenAI
    default_model = "gpt-4o-mini"

    def import_client(self):
        from openai import OpenAI
        return OpenAI
    
    
    def prep_input_message(self, message: Message) -> dict:
        message_dict = {
            "role": message.role,
            "content": message.content,                
        }
        if message.tool_calls:
            if message.role == "tool":
                tool_call = message.tool_calls[0]
                message_dict["tool_call_id"] = tool_call.tool_call_id
            else:
                message_dict["tool_calls"] = [
                    self.prep_input_tool_call(tool_call) for tool_call in message.tool_calls
                ]

        if message.images:
            # TODO: Implement image handling
            message_dict["images"] = message.images

        return message_dict
    
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

    
    def prep_input_tool(self, tool: Tool) -> dict:
        return tool.to_dict()
    
    def prep_input_tool_choice(self, tool_choice: str) -> Union[str, dict]:
        if tool_choice in ("auto", "required", "none"):
            return tool_choice

        tool_type = "function" # Currently only function tools are supported See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        return {"type": tool_type, tool_type: {"name": tool_choice}}

    def prep_input_tool_call(self, tool_call: ToolCall) -> dict:
        return {
            "id": tool_call.tool_call_id,
            "type": tool_call.type,
            tool_call.type: {
                "name": tool_call.tool_name,
                "arguments": stringify_content(tool_call.arguments),
            }
        }


    def split_tool_call_outputs_into_messages(self, tool_calls: list[ToolCall]) -> list[Message]:        
        return [
            Message(role="tool", content=stringify_content(tool_call.output), tool_calls=[tool_call])
            for tool_call in tool_calls
        ]
    
    # def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> dict:
    #     return {
    #         "role": "tool",
    #         "tool_call_id": tool_call.tool_call_id,
    #         "name": tool_call.tool_name,
    #         "content": tool_response,
    #     }

    def prep_input_response_format(self, response_format: Union[str, dict]) -> Union[str, dict]:
        simple_response_formats = ("json", "json_object", "text")
        
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
        

    def extract_output_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> ToolCall:
        return ToolCall(
            tool_call_id=tool_call.id,
            tool_name=tool_call.function.name,
            arguments=json_loads(tool_call.function.arguments),
        )    
    
    
    # def extract_output_message(self, response: ChatCompletion) -> Message:
    #     message = response.choices[0].message
    #     images = None # TODO: Implement image extraction
    #     tool_calls = [self.extract_output_tool_call(tool_call) for tool_call in message.tool_calls] if message.tool_calls else None
    #     return Message(
    #         role=message.role,
    #         content=message.content,
    #         images=images,
    #         tool_calls=tool_calls,
    #         response_object=response,
    #     )

    def extract_std_and_client_messages(self, response: ChatCompletion) -> tuple[Message, ChatCompletionMessage]:
        client_message = response.choices[0].message
        images = None # TODO: Implement image extraction
        tool_calls = [self.extract_output_tool_call(tool_call) for tool_call in client_message.tool_calls] if client_message.tool_calls else None
        std_message = Message(
            role=client_message.role,
            content=client_message.content,
            images=images,
            tool_calls=tool_calls,
            response_object=response,
        )   
        return std_message, client_message


    def list_models(self) -> list[str]:
        return [model.id for model in self.client.models.list()]

    def chat(
            self,
            messages: list[dict],     
            model: str = default_model,
            system_prompt: Optional[str] = None,                    
            tools: Optional[list[dict]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
            response_format: Optional[str] = None,
            **kwargs
            ) -> ChatCompletion:

            if tool_choice and not tools:
                tool_choice = None

            response = self.client.chat.completions.create(
                messages=messages, 
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                **kwargs
            )
            return response


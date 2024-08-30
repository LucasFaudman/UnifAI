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
from .baseaiclientwrapper import BaseAIClientWrapper

class OpenAIWrapper(BaseAIClientWrapper):
    client: OpenAI
    default_model = "gpt-4o-mini"

    def import_client(self):
        from openai import OpenAI
        return OpenAI
    
    def prep_input_message(self, message: Union[Message, dict[str, str], str]) -> dict:
        if isinstance(message, str):
            return {"role": "user", "content": message}        
        elif isinstance(message, Message):
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
                        {
                            "id": tool_call.tool_call_id,
                            "type": tool_call.type,
                            tool_call.type: {
                                "name": tool_call.tool_name,
                                "arguments": self.format_content(tool_call.arguments),
                            },                            
                        }
                        for tool_call in message.tool_calls
                    ]

            if message.images:
                message_dict["images"] = message.images

            return message_dict
        else:
            # message is a dict
            return message
    
    def prep_input_tool(self, tool: Union[Tool, dict]) -> dict:
        if isinstance(tool, Tool):
            return tool.to_dict()
        return tool
    
    def prep_input_tool_choice(self, tool_choice: Union[Tool, str, dict, Literal["auto", "required", "none"]]) -> Union[str, dict]:
        if tool_choice in ("auto", "required", "none"):
            return tool_choice
        if isinstance(tool_choice, Tool):
            tool_type = tool_choice.type
            tool_name = tool_choice.name
        else:
            tool_type = "function"
            tool_name = tool_choice
        return {"type": tool_type, tool_type: {"name": tool_name}}
    
    def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call.tool_call_id,
            "name": tool_call.tool_name,
            "content": tool_response,
        }

    def prep_input_response_format(self, response_format: Union[str, dict[str, str]]) -> Union[str, dict[str, str]]:
        if response_format in ("json", "json_object", "text"):
            return {"type": response_format}
        return response_format
    
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

    # def chat(
    #         self,
    #         messages: list[dict],     
    #         model: str = default_model,                    
    #         tools: Optional[list[dict]] = None,
    #         tool_choice: Optional[Union[Literal["auto", "required", "none"], dict]] = None,
    #         response_format: Optional[str] = None,
    #         **kwargs
    #         ) -> Message:

    #         model = model or self.default_model        
    #         messages = [self.prep_input_message(message) for message in ([] if messages is None else messages)]
    #         tools = [self.prep_input_tool(tool) for tool in tools] if tools else None
    #         tool_choice = self.prep_input_tool_choice(tool_choice) if tool_choice else None
    #         response_format = self.prep_input_response_format(response_format) if response_format else None

    #         response = self.client.chat.completions.create(
    #             messages=messages, 
    #             model=model,
    #             tools=tools,
    #             tool_choice=tool_choice,
    #             response_format=response_format,
    #             **kwargs
    #         )
    #         return self.extract_output_message(response)

    def chat(
            self,
            messages: list[dict],     
            model: str = default_model,                    
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


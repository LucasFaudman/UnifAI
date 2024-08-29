from typing import Optional, Union, Sequence, Any, Literal, Mapping

from ._types import Message, Tool, ToolCall
from .baseaiclientwrapper import BaseAIClientWrapper

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
    Options as OllamaOptions
)


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
    
    def prep_input_message(self, message: Union[Message, dict[str, str], str, OllamaMessage]) -> OllamaMessage:
        role = 'user'
        content = None
        images = None
        tool_calls = None
        
        if isinstance(message, str):
            role = 'user'
            content = message
        elif isinstance(message, Message):
            role = message.role
            content = message.content
            if message.images:
                images = [image.data for image in message.images if image.data]
            if message.tool_calls:
                if message.role == 'tool':
                    tool_calls = None
                else:
                    tool_calls = [self.prep_input_tool_call(tool_call) for tool_call in message.tool_calls]
            
        elif isinstance(message, dict):
            role = message['role']
            content = message.get('content')
            images = message.get('images', None)
            tool_calls = message.get('tool_calls', None)

        return OllamaMessage(role=role, content=content, images=images, tool_calls=tool_calls)
    
    def prep_input_tool(self, tool: Union[Tool, dict]) -> OllamaTool:
        tool_dict = tool if isinstance(tool, dict) else tool.to_dict()

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
    
    def prep_input_tool_choice(self, tool_choice: Union[Tool, str, dict, Literal["auto", "required", "none"]]) -> str:
        # if tool_choice in ("auto", "required", "none"):
        #     return tool_choice
        if isinstance(tool_choice, Tool):
            return tool_choice.name
        if isinstance(tool_choice, dict):
            tool_type = tool_choice['type']
            return tool_choice[tool_type]['name']
        return tool_choice
    
    def prep_input_tool_call(self, tool_call: ToolCall) -> OllamaToolCall:
        return OllamaToolCall(
            function=OllamaToolCallFunction(
                name=tool_call.tool_name, 
                arguments=tool_call.arguments if tool_call.arguments else {}
            )
        )
    
    def prep_input_tool_call_response(self, tool_call: ToolCall, tool_response: Any) -> OllamaMessage:
        return OllamaMessage(
            role='tool',
            content=tool_response,
        )

    def prep_input_response_format(self, response_format: Union[str, dict[str, str]]) -> str:
        if response_format in ("json", "json_object"):
            return 'json'
        return ''

    def extract_output_tool_call(self, tool_call: OllamaToolCall) -> ToolCall:
        tool_call_function = tool_call['function']
        name = tool_call_function['name']
        arguments = tool_call_function.get('arguments')
        tool_call_id = f'call_{generate_random_id(24)}'
        return ToolCall(tool_call_id=tool_call_id, tool_name=name, arguments=arguments)


    def extract_output_message(self, response: OllamaChatResponse) -> Message:
        message = response['message']
        images = None # TODO: Implement image extraction
        tool_calls = [self.extract_output_tool_call(tool_call) for tool_call in message.get('tool_calls', [])]
        return Message(
            role=message['role'],
            content=message.get('content'),
            images=images,
            tool_calls=tool_calls,
            response_object=response
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
            messages: list[Message],     
            model: Optional[str] = None,                    
            tools: Optional[list[Any]] = None,
            tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,
            **kwargs
            ) -> Message:
        
            # model = model or self.default_model
            # if system_prompt := self.get_system_prompt_from_messages(messages):
            #     model = self.get_custom_model(model, system_prompt)
            #     messages = messages[1:] # Remove system prompt message

            messages = [self.prep_input_message(message) for message in ([] if messages is None else messages)]
            tools = [self.prep_input_tool(tool) for tool in tools] if tools else None
            tool_choice = self.prep_input_tool_choice(tool_choice) if tool_choice else None
            response_format = self.prep_input_response_format(response_format) if response_format else None

            keep_alive = kwargs.pop('keep_alive', None)
            stream = kwargs.pop('stream', False)
            options = kwargs.pop('options', None)
            if not options and kwargs:
                options = OllamaOptions(**kwargs)

            response = self.client.chat(
                messages=messages,                 
                model=model, 
                tools=tools, 
                format=response_format, 
                keep_alive=keep_alive,
                stream=stream,
                options=options
            )
            return self.extract_output_message(response)
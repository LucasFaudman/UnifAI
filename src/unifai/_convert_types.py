from json import dumps as json_dumps
from typing import Optional, Union, Sequence, Any, Literal, Mapping, TypeVar, List, Type, Tuple, Dict, Any
from ._types import (
    Message,
    Tool,
    FunctionTool,
    CodeInterpreterTool,
    FileSearchTool,
    ToolCall,
    ToolParameter, 
    ToolDict, 
    ToolParameterType, 
    ToolValPyTypes, 
    StringToolParameter, 
    NumberToolParameter, 
    IntegerToolParameter,
    BooleanToolParameter,
    NullToolParameter,
    ArrayToolParameter,
    ObjectToolParameter,
    AnyOfToolParameter,
)

def make_content_serializeable(content: Any) -> Union[str, int, float, bool, dict, list, None]:
    """Recursively makes an object serializeable by converting it to a dict or list of dicts and converting all non-string values to strings."""
    if content is None or isinstance(content, (str, int, float, bool)):
        return content
    if isinstance(content, Mapping):
        return {k: make_content_serializeable(v) for k, v in content.items()}
    if isinstance(content, Sequence):
        return [make_content_serializeable(item) for item in content]
    return str(content) 

def stringify_content(content: Any) -> str:
    """Formats content for use a message content. If content is not a string, it is converted to a json string."""
    if isinstance(content, str):
        return content
    return json_dumps(make_content_serializeable(content), indent=0)

def make_few_shot_prompt(        
        system_prompt: Optional[str] = None, 
        examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None, 
        content: Any = ""
    ) -> Sequence[Message]:
    """Makes list of message objects from system prompt, examples, and user input."""
    messages = []
    if system_prompt:
        # Begin with system_prompt if it exists
        messages.append(Message(role="system", content=system_prompt))

    # Add examples
    if examples:
        for example in examples:
            if isinstance(example, Message):
                messages.append(example)
            else:
                messages.append(Message(role="user", content=stringify_content(example['input'])))
                messages.append(Message(role="assistant", content=stringify_content(example['response'])))

    # Add content
    messages.append(Message(role="user", content=stringify_content(content)))
    return messages


def tool_parameter_from_dict(
        param_dict: dict, 
        param_name: Optional[str]= None,
        param_required: bool= False
        ) -> ToolParameter:
    
    if (anyof_param_dicts := param_dict.get('anyOf')) is not None:
        anyOf = [
            tool_parameter_from_dict(param_dict=anyof_param_dict, param_name=param_name, param_required=param_required)
            for anyof_param_dict in anyof_param_dicts
        ]
        return AnyOfToolParameter(name=param_name, required=param_required, anyOf=anyOf)

    param_type = param_dict['type']
    param_name = param_dict.get('name', param_name)
    param_description = param_dict.get('description')
    param_enum = param_dict.get('enum')

    if param_type == 'string':
        return StringToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'number':
        return NumberToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'integer':
        return IntegerToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'boolean':
        return BooleanToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'null':
        return NullToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    
    if param_type == 'array':
        items = tool_parameter_from_dict(param_dict=param_dict['items'], param_required=param_required)
        return ArrayToolParameter(name=param_name, description=param_description, 
                                  required=param_required, enum=param_enum, 
                                  items=items)    
    if param_type == 'object':
        required_params = param_dict.get('required', [])
        properties = [
            tool_parameter_from_dict(param_dict=prop_dict, param_name=prop_name, param_required=prop_name in required_params) 
            for prop_name, prop_dict in param_dict['properties'].items()
        ]
        additionalProperties = param_dict.get('additionalProperties', False)
        return ObjectToolParameter(name=param_name, description=param_description, 
                                   required=param_required, enum=param_enum, 
                                   properties=properties, additionalProperties=additionalProperties)
    
    raise ValueError(f"Invalid parameter type: {param_type}")



def tool_from_dict(tool_dict: dict) -> Tool:
    tool_type = tool_dict['type']
    if tool_type == 'code_interpreter':
        return CodeInterpreterTool()
    if tool_type == 'file_search':
        return FileSearchTool()
    
    tool_def = tool_dict[tool_type]
    parameters = tool_parameter_from_dict(param_dict=tool_def['parameters'], 
            # param_name='parameters',
            # param_required=True
    )
    if parameters.type == 'anyOf':
        raise ValueError("Root parameter cannot be anyOf: See: https://platform.openai.com/docs/guides/structured-outputs/root-objects-must-not-be-anyof")

    return FunctionTool(
        name=tool_def['name'], 
        description=tool_def['description'], 
        parameters=parameters,
        strict=tool_def.get('strict', True)
    )
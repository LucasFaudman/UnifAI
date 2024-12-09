from ...types import (
    ObjectToolParameter,
    AnyOfToolParameter,
    Tool,
    PROVIDER_TOOLS,    
)

from .construct_tool_parameter import construct_tool_parameter

def tool_from_dict(tool_dict: dict) -> Tool:
    tool_type = tool_dict['type']
    if provider_tool := PROVIDER_TOOLS.get(tool_type):
        return provider_tool

    tool_def = tool_dict.get(tool_type) or tool_dict.get("input_schema")
    if tool_def is None:
        raise ValueError("Invalid tool definition. "
                         f"The input schema must be defined under the key '{tool_type}' or 'input_schema' when tool type='{tool_type}'.")

    parameters = construct_tool_parameter(param_dict=tool_def['parameters'])
    if isinstance(parameters, AnyOfToolParameter):
        raise ValueError("Root parameter cannot be anyOf: See: https://platform.openai.com/docs/guides/structured-outputs/root-objects-must-not-be-anyof")    
    if not isinstance(parameters, ObjectToolParameter):
        raise ValueError("Root parameter must be an object")
    
    return Tool(
        name=tool_def['name'], 
        description=tool_def['description'], 
        parameters=parameters,
        strict=tool_def.get('strict', True)
    )
from typing import Optional
from unifai.types import (
    ToolParameter,
    StringToolParameter,
    NumberToolParameter,
    IntegerToolParameter,
    BooleanToolParameter,
    NullToolParameter,
    ArrayToolParameter,
    ObjectToolParameter,
    AnyOfToolParameter,
    Tool,
    ProviderTool,
    PROVIDER_TOOLS,
    
)


def tool_parameter_from_dict(
        param_dict: dict, 
        param_name: Optional[str]= None,
        param_required: bool= True
        ) -> ToolParameter:
    
    # if isinstance(required := param_dict.get('required'), bool):
    #     param_required = required

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
    # param_required = param_dict.get('required', param_required)

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
        if not (param_items := param_dict.get('items')):
            raise ValueError("Array parameters must have an 'items' key.")
        
        items = tool_parameter_from_dict(param_dict=param_items, param_required=param_required)
        return ArrayToolParameter(name=param_name, description=param_description, 
                                  required=param_required, enum=param_enum, 
                                  items=items)    
    if param_type == 'object':
        if not (param_properties := param_dict.get('properties')):
            raise ValueError("Object parameters must have a 'properties' key.")
        if isinstance(param_properties, dict):
            required_params = param_dict.get('required', [])
            properties = [
                tool_parameter_from_dict(param_dict=prop_dict, param_name=prop_name, param_required=prop_name in required_params) 
                for prop_name, prop_dict in param_dict['properties'].items()
            ]
        else:
            properties = [
                tool_parameter_from_dict(param_dict=prop_dict, param_required=prop_dict.get('required', True)) 
                for prop_dict in param_properties
            ]
        additionalProperties = param_dict.get('additionalProperties', False)
        return ObjectToolParameter(name=param_name, description=param_description, 
                                   required=param_required, enum=param_enum, 
                                   properties=properties, additionalProperties=additionalProperties)
    
    raise ValueError(f"Invalid parameter type: {param_type}")


def tool_from_dict(tool_dict: dict) -> Tool:
    tool_type = tool_dict['type']
    if provider_tool := PROVIDER_TOOLS.get(tool_type):
        return provider_tool

    tool_def = tool_dict.get(tool_type) or tool_dict.get("input_schema")
    if tool_def is None:
        raise ValueError("Invalid tool definition. "
                         f"The input schema must be defined under the key '{tool_type}' or 'input_schema' when tool type='{tool_type}'.")

    parameters = tool_parameter_from_dict(param_dict=tool_def['parameters'], 
            # param_name='parameters',
            # param_required=True
    )
    if not isinstance(parameters, ObjectToolParameter):
        raise ValueError("Root parameter must be an object")
    
    # if isinstance(parameters, AnyOfToolParameter):
    #     raise ValueError("Root parameter cannot be anyOf: See: https://platform.openai.com/docs/guides/structured-outputs/root-objects-must-not-be-anyof")

    return Tool(
        name=tool_def['name'], 
        description=tool_def['description'], 
        parameters=parameters,
        strict=tool_def.get('strict', True)
    )
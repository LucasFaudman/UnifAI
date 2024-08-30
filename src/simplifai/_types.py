from typing import Optional, Union, Sequence, Any, Literal, Mapping, TypeVar, List, Type, Tuple, Dict, Any

from pydantic import BaseModel, Field

class ResponseInfo(BaseModel):
    data: Any
    usage: Any

class Image(BaseModel):
    url: Optional[str]
    data: Optional[Union[str, bytes]]
    filepath: Optional[str]

class ToolCall(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: Optional[Mapping[str, Any]]
    type: str = "function"


class Message(BaseModel):
    role: Literal['user', 'assistant', 'system', 'tool']
    content: Optional[str] = None
    images: Optional[list[Image]] = None
    tool_calls: Optional[list[ToolCall]] = None
    response_object: Optional[Any] = None
    # response_info: Optional[ResponseInfo]


ToolParameterType = Literal["object", "array", "string", "integer", "number", "boolean", "null"]
ToolValPyTypes = Union[str, int, float, bool, None, list, dict]
ToolDict = dict[str, ToolValPyTypes]


class ToolParameter(BaseModel):
    type: ToolParameterType = "string"
    name: Optional[str] = None
    description: Optional[str] = None
    enum: Optional[list[ToolValPyTypes]] = None
    required: bool = False
    # enum: list[str] = Field(default_factory=list)
    

    def to_dict(self) -> ToolDict:
        param_dict: ToolDict = {"type": self.type}
        if self.description:
            param_dict["description"] = self.description
        if self.enum:
            param_dict["enum"] = self.enum
        return param_dict
    
class StringToolParameter(ToolParameter):
    type: Literal["string"] = "string"

class NumberToolParameter(ToolParameter):
    type: Literal["number"] = "number"

class IntegerToolParameter(ToolParameter):
    type: Literal["integer"] = "integer"

class BooleanToolParameter(ToolParameter):
    type: Literal["boolean"] = "boolean"

class NullToolParameter(ToolParameter):
    type: Literal["null"] = "null"

class ArrayToolParameter(ToolParameter):
    type: Literal["array"] = "array"
    items: ToolParameter
    
    def to_dict(self) -> ToolDict:
        return {
            **ToolParameter.to_dict(self),
            "items": self.items.to_dict() 
        }
    
class ObjectToolParameter(ToolParameter):
    type: Literal["object"] = "object"
    properties: list[ToolParameter]
    additionalProperties: bool = False
    
    def to_dict(self) -> ToolDict:
        properties = {}
        required = []
        for prop in self.properties:
            properties[prop.name] = prop.to_dict()
            if prop.required:
                required.append(prop.name)

        return { 
            **ToolParameter.to_dict(self),
            "properties": properties,
            "required": required,
            "additionalProperties": self.additionalProperties
        }
    
class AnyOfToolParameter(ToolParameter):
    type: Literal["anyOf"] = "anyOf"
    anyOf: list[ToolParameter]

    def to_dict(self) -> ToolDict:
        return {
            "anyOf": [param.to_dict() for param in self.anyOf]
        }

    
class Tool(BaseModel):
    type: str
    name: str

    def to_dict(self):
        return {
            "type": self.type,
            # self.type: {
            #     "name": self.name
            # }
        }

class FunctionTool(Tool):
    type: Literal["function"] = "function"
    description: str
    parameters: Union[ObjectToolParameter, ArrayToolParameter, ToolParameter]
    strict: bool = True

    def to_dict(self):
        return {
            "type": self.type,
            self.type: {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_dict(),
                "strict": self.strict
            },            
        }

class CodeInterpreterTool(Tool):
    type: Literal["code_interpreter"] = "code_interpreter"
    name: str = "code_interpreter"

class FileSearchTool(Tool):
    type: Literal["file_search"] = "file_search"
    name: str = "file_search"


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
        strict=tool_def.get('strict')
    )
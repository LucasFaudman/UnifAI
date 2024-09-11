from typing import Optional, Union, Sequence, Any, Literal
from pydantic import BaseModel


ToolParameterType = Literal["object", "array", "string", "integer", "number", "boolean", "null"]
ToolValPyTypes = Union[str, int, float, bool, None, list[Any], dict[str, Any]]
# ToolDict = dict[str, ToolValPyTypes]


class ToolParameter(BaseModel):
    type: ToolParameterType = "string"
    name: Optional[str] = None
    description: Optional[str] = None
    enum: Optional[list[ToolValPyTypes]] = None
    required: bool = True
        
    def to_dict(self) -> dict:
        param_dict: dict = {"type": self.type}
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
    
    def to_dict(self) -> dict:
        return {
            **ToolParameter.to_dict(self),
            "items": self.items.to_dict() 
        }
    

class ObjectToolParameter(ToolParameter):
    type: Literal["object"] = "object"
    properties: Sequence[ToolParameter]
    additionalProperties: bool = False
    
    def to_dict(self) -> dict:
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

    def to_dict(self) -> dict:
        return {
            "anyOf": [param.to_dict() for param in self.anyOf]
        }


class ToolParameters(ObjectToolParameter):
    def __init__(self, *parameters: ToolParameter, **kwargs):
        super().__init__(properties=parameters, **kwargs)
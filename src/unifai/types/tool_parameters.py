from typing import Optional, Union, Sequence, Any, Literal, Mapping, Collection
from pydantic import BaseModel


ToolParameterType = Literal["object", "array", "string", "integer", "number", "boolean", "null"]
ToolParameterPyTypes = Union[str, int, float, bool, None, list[Any], dict[str, Any]]
ToolParameterExcludableKeys = Literal["description", "enum", "required", "additionalProperties", "defs", "refs"]
EXCLUDE_NONE = frozenset()

class ToolParameter(BaseModel):
    type: ToolParameterType = "string"
    name: Optional[str] = None
    description: Optional[str] = None
    enum: Optional[list[ToolParameterPyTypes]] = None
    required: bool = True
        
    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        param_dict: dict = {"type": self.type}
        if self.description and "description" not in exclude:
            param_dict["description"] = self.description
        if self.enum and "enum" not in exclude:
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


class RefToolParameter(ToolParameter):
    type: Literal["ref"] = "ref"
    ref: str

    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        return {"$ref": self.ref} if "ref" not in exclude else {}


class ArrayToolParameter(ToolParameter):
    type: Literal["array"] = "array"
    items: ToolParameter
    
    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        return {
            **ToolParameter.to_dict(self, exclude),
            "items": self.items.to_dict(exclude) 
        }
    

class ObjectToolParameter(ToolParameter):
    type: Literal["object"] = "object"
    properties: Sequence[ToolParameter]
    additionalProperties: bool = False
    defs: Optional[Mapping[str, ToolParameter]] = None
    
    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        properties = {}
        required = []
        for prop in self.properties:
            properties[prop.name] = prop.to_dict(exclude)
            if prop.required:
                required.append(prop.name)

        self_dict = { 
            **ToolParameter.to_dict(self, exclude),
            "properties": properties,
        }
        if "required" not in exclude:
            self_dict["required"] = required
        if "additionalProperties" not in exclude:
            self_dict["additionalProperties"] = self.additionalProperties

        if self.defs and "defs" not in exclude:
            self_dict["$defs"] = {name: prop.to_dict(exclude) for name, prop in self.defs.items()}
        
        return self_dict

    
class AnyOfToolParameter(ToolParameter):
    type: Literal["anyOf"] = "anyOf"
    anyOf: list[ToolParameter]

    def __init__(self, 
                 name: str, 
                 anyOf: list[ToolParameter], 
                 description: Optional[str] = None,
                 required: bool = True, 
                 **kwargs):
        
        for tool_parameter in anyOf:
            if not tool_parameter.name:
                tool_parameter.name = name
        
        BaseModel.__init__(self, name=name, description=description, required=required, anyOf=anyOf, **kwargs)

    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        return {
            "anyOf": [param.to_dict(exclude) for param in self.anyOf]
        }


class ToolParameters(ObjectToolParameter):
    def __init__(self, *parameters: ToolParameter, **kwargs):
        super().__init__(properties=parameters, **kwargs)
from typing import Optional, Union, Sequence, Any, Literal, Mapping, Collection
from ._base_model import BaseModel
from .annotations import DefaultT

ToolParameterType = Literal["object", "array", "string", "integer", "number", "boolean", "null"]
ToolParameterPyTypes = Union[str, int, float, bool, None, list[Any], dict[str, Any]]
ToolParameterExcludableKeys = Literal["description", "enum", "required", "additionalProperties", "defs", "refs", "strict"]
EXCLUDE_NONE = frozenset[ToolParameterExcludableKeys]()

class ToolParameter(BaseModel):
    type: ToolParameterType = "string"
    name: Optional[str] = None
    description: Optional[str] = None
    enum: Optional[list[ToolParameterPyTypes]] = None
        
    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        self_dict: dict = {"type": self.type}
        if self.description and "description" not in exclude:
            self_dict["description"] = self.description
        if self.enum and "enum" not in exclude:
            self_dict["enum"] = self.enum
        return self_dict


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

    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        return {"type": "null"}


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
    properties: dict[str, ToolParameter]
    additionalProperties: bool = False
    defs: Optional[dict[str, ToolParameter]] = None

    def _update_param_dict(
            self, 
            param_dict: dict[str, ToolParameter], 
            *args: ToolParameter,
            **kwargs: ToolParameter
        ) -> dict[str, ToolParameter]:
        for arg in args:
            if not arg.name:
                raise ValueError("All properties must have a name when passed as a sequence")
            param_dict[arg.name] = arg        
        for param_name, param in kwargs.items():
            if not param.name:
                param.name = param_name
            param_dict[param_name] = param
        return param_dict

    def __init__(self, 
                 properties: dict[str, ToolParameter] | Sequence[ToolParameter], 
                 additionalProperties: bool = False,
                 required: list[str] | Literal["all"] = "all",
                 defs: Optional[dict[str, ToolParameter] | Sequence[ToolParameter]] = None, 
                 **kwargs
                 ):        
        if isinstance(properties, Mapping):
            kwargs["properties"] = self._update_param_dict({}, **properties)
        else:
            kwargs["properties"] = self._update_param_dict({}, *properties)
        kwargs["additionalProperties"] = additionalProperties

        if defs:
            if isinstance(defs, Mapping):
                kwargs["defs"] = self._update_param_dict({}, **defs)
            else:
                kwargs["defs"] = self._update_param_dict({}, *defs)

        BaseModel.__init__(self, **kwargs)
        self._required = required
    
    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        properties = {prop_name: prop.to_dict(exclude) for prop_name, prop in self.properties.items()}
        self_dict = { 
            **ToolParameter.to_dict(self, exclude),
            "properties": properties,
        }
        if "required" not in exclude:
            self_dict["required"] = self.required
        if "additionalProperties" not in exclude:
            self_dict["additionalProperties"] = self.additionalProperties
        if self.defs and "defs" not in exclude:
            self_dict["$defs"] = {name: prop.to_dict(exclude) for name, prop in self.defs.items()}
                    
        return self_dict
    
    @property
    def required(self) -> list[str]:
        if isinstance(self._required, list):
            return self._required
        return list(self.properties.keys())
    
    @required.setter
    def required(self, value: list[str] | Literal["all"]) -> None:
        self._required = value
    
    # Properties (dict-like) methods
    def __len__(self) -> int:
        return len(self.properties)

    def __contains__(self, key: str) -> bool:
        return key in self.properties
    
    def __iter__(self):
        return iter(self.properties)
    
    def keys(self):
        return self.properties.keys()
    
    def values(self):
        return self.properties.values()
    
    def items(self):
        return self.properties.items()
    
    def __getitem__(self, key: str) -> ToolParameter:
        return self.properties[key]
    
    def get(self, key: str, default: DefaultT = None) -> ToolParameter | DefaultT:
        return self.properties.get(key, default)

    def __setitem__(self, key: str, value: ToolParameter) -> None:
        self.properties[key] = value

    def setdefault(self, key: str, default: ToolParameter) -> ToolParameter:
        return self.properties.setdefault(key, default)        
    
    def update(self, *args: ToolParameter, **kwargs: ToolParameter) -> None:
        self._update_param_dict(self.properties, *args, **kwargs)    

    def __delitem__(self, key: str) -> None:
        del self.properties[key]

    def pop(self, key: str, default: DefaultT = None) -> ToolParameter | DefaultT:
        return self.properties.pop(key, default)

    def clear(self) -> None:
        self.properties.clear()

    # $def/$ref methods
    def get_def(self, key: str, default: DefaultT = None) -> ToolParameter | DefaultT:
        return self.defs.get(key, default) if self.defs else default
    
    def set_def(self, key: str, value: ToolParameter) -> None:
        if not self.defs:
            self.defs = {}
        self.defs[key] = value

    def update_defs(self, *args: ToolParameter, **kwargs: ToolParameter) -> None:
        if self.defs is None:
            self.defs = {}
        self._update_param_dict(self.defs, *args, **kwargs)

    def pop_def(self, key: str, default: DefaultT = None) -> ToolParameter | DefaultT:
        return self.defs.pop(key, default) if self.defs else default
    
    def clear_defs(self) -> None:
        if self.defs:
            self.defs.clear()

    
class AnyOfToolParameter(ToolParameter):
    type: Literal["anyOf"] = "anyOf"
    anyOf: list[ToolParameter]

    def __init__(self, 
                 name: Optional[str], 
                 anyOf: list[ToolParameter], 
                 description: Optional[str] = None,
                 **kwargs):

        # Set name for all parameters in anyOf to be same as self if not set        
        for tool_parameter in anyOf:
            if not tool_parameter.name:
                tool_parameter.name = name

        kwargs["name"] = name
        kwargs["description"] = description
        kwargs["anyOf"] = anyOf        
        BaseModel.__init__(self, **kwargs)

    def to_dict(self, exclude: Collection[ToolParameterExcludableKeys] = EXCLUDE_NONE) -> dict:
        return {
            "anyOf": [param.to_dict(exclude) for param in self.anyOf]
        }


class OptionalToolParameter(AnyOfToolParameter):
    def __init__(self, tool_parameter: ToolParameter):
        super().__init__(name=tool_parameter.name, anyOf=[tool_parameter, NullToolParameter()])
    


class ToolParameters(ObjectToolParameter):
    def __init__(self, *parameters: ToolParameter, **kwargs):
        super().__init__(properties=parameters, **kwargs)
from typing import (
    Optional, Union, Sequence, Any, 
    Literal, Mapping, TypeVar, List, Type, Tuple, Dict, Any, Callable, Collection)

from pydantic import BaseModel, Field
from datetime import datetime

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    # total_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class ResponseInfo(BaseModel):
    model: Optional[str] = None    
    # done: bool
    done_reason: Optional[Literal["stop", "tool_calls", "max_tokens", "content_filter"]] = None
    usage: Optional[Usage] 
    # duration: Optional[int]
    # created_at: datetime = Field(default_factory=datetime.now)
    


class Image(BaseModel):    
    data: Optional[Union[str, bytes]]
    url: Optional[str]
    filepath: Optional[str]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = "image/jpeg"
    format: Literal["base64", "url", "filepath"] = "base64"


class ToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: Mapping[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    type: str = "function"


class Message(BaseModel):
    # id: str
    role: Literal['user', 'assistant', 'tool', 'system']
    content: Optional[str] = None
    images: Optional[list[Image]] = None
    tool_calls: Optional[list[ToolCall]] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    response_info: Optional[ResponseInfo] = None


ToolParameterType = Literal["object", "array", "string", "integer", "number", "boolean", "null"]
ToolValPyTypes = Union[str, int, float, bool, None, list[Any], dict[str, Any]]
ToolDict = dict[str, ToolValPyTypes]


class ToolParameter(BaseModel):
    type: ToolParameterType = "string"
    name: Optional[str] = None
    description: Optional[str] = None
    enum: Optional[list[ToolValPyTypes]] = None
    required: bool = True
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
    properties: Sequence[ToolParameter]
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

class ToolParameters(ObjectToolParameter):
    def __init__(self, *parameters: ToolParameter, **kwargs):
        super().__init__(properties=parameters, **kwargs)

    
class Tool(BaseModel):
    type: str
    name: str
    callable: Optional[Callable] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
        }


class FunctionTool(Tool):
    type: Literal["function"] = "function"
    description: str
    parameters: Union[ObjectToolParameter, ArrayToolParameter, ToolParameter]
    strict: bool = True
    callable: Optional[Callable] = None


    def __init__(self, 
        name: str, 
        description: str, 
        *args: ToolParameter,
        parameters: Optional[Union[ObjectToolParameter, Sequence[ToolParameter], Mapping[str, ToolParameter]]] = None,
        type: str = "function",
        strict: bool = True,
        callable: Optional[Callable] = None
    ):        
             
        if isinstance(parameters, ObjectToolParameter):
            parameters = parameters
            # parameters.name = "parameters"
        elif isinstance(parameters, Sequence):
            parameters = ObjectToolParameter(properties=parameters)
        elif isinstance(parameters, Mapping):
            properties = []
            for parameter_name, parameter in parameters.items():
                if parameter.name is None:
                    parameter.name = parameter_name
                elif parameter.name != parameter_name:
                    raise ValueError("Parameter name does not match key")
                properties.append(parameter)
            parameters = ObjectToolParameter(properties=properties)
        elif args:
            parameters = ObjectToolParameter(properties=list(args))
        else:
            raise ValueError(f"Invalid parameters type: {parameters}")

        BaseModel.__init__(self, name=name, type=type, description=description, parameters=parameters, strict=strict, callable=callable)


    def __call__(self, *args, **kwargs) -> Any:
        if self.callable is None:
            raise ValueError(f"Callable not set for tool {self.name}")
        return self.callable(*args, **kwargs)


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


class EvaluateParameters(BaseModel):
    eval_type: str
    system_prompt: str = "Your role is to evaluate the content using the provided tool(s)." 
    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None   
    tools: Optional[list[Union[Tool, str]]] = None
    tool_choice: Optional[Union[str, list[str]]] = None
    return_on: Optional[Union[Literal["content", "tool_call", "message"], str, list[str]]] = None
    return_as: Literal["chat", 
                       "messages", 
                       "last_message", 
                       "last_content",
                       "last_tool_call",
                       "last_tool_call_args",
                       "last_tool_calls", 
                       "last_tool_calls_args"
                       ] = "chat"

    
    response_format: Optional[Union[str, dict[str, str]]] = None
    enforce_tool_choice: bool = True
    tool_choice_error_retries: int = 3


MessageInput = Sequence[Union[Message, str, dict[str, Any]]]
ToolInput = Union[Tool, dict[str, Any], str]
EvaluateParametersInput = Union[EvaluateParameters, dict[str, Any]]
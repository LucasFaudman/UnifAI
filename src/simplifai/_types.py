from typing import (
    Optional, Union, Sequence, Any, 
    Literal, Mapping, TypeVar, List, Type, Tuple, Dict, Any, Callable, Collection)

from pydantic import BaseModel, Field

class ResponseInfo(BaseModel):
    data: Any
    usage: Any

class Image(BaseModel):    
    data: Optional[Union[str, bytes]]
    url: Optional[str]
    filepath: Optional[str]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = "image/jpeg"
    format: Literal["base64", "url", "filepath"] = "base64"

class ToolCall(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: Optional[Mapping[str, Any]]
    output: Optional[Any] = None
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

    def to_dict(self) -> dict:
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
    callable: Optional[Callable] = None

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


class EvalTypeParameters(BaseModel):
    name: str
    system_prompt: str = "Your role is to evaluate the content using the provided tool(s)." 
    examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None   
    tools: Optional[list[Union[Tool, str]]] = None
    tool_choice: Optional[Union[str, list[str]]] = None
    return_on: Optional[Union[Literal["content", "tool_call", "message"], str, list[str]]] = None
    return_as: Literal["messages", "last_message", "last_content_or_tool_call_args", "last_content_or_all_tool_call_args"] = "last_content_or_tool_call_args"

    
    response_format: Optional[Union[str, dict[str, str]]] = None
    enforce_tool_choice: bool = True
    tool_choice_error_retries: int = 3


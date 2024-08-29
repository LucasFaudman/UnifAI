from typing import Optional, Union, Sequence, Any, Literal, Mapping

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


ToolParameterType = Literal["object", "array", "string", "number", "boolean", "null"]

class ToolParameter(BaseModel):    
    name: str
    description: str
    type: ToolParameterType = "string"
    required: bool = False
    options: list[str] = Field(default_factory=list)

    def to_dict(self):
        return { 
            "type": self.type, 
            "description": self.description 
        }
    
class ToolObjectParameter(ToolParameter):
    type: Literal["object"] = "object"
    properties: list[ToolParameter]
    
    def to_dict(self):
        properties = {}
        required = []
        for prop in self.properties:
            properties[prop.name] = prop.to_dict()
            if prop.required:
                required.append(prop.name)

        return { 
            **ToolParameter.to_dict(self),
            "properties": properties,
            "required": required
        }
    
class ToolArrayParameter(ToolParameter):
    type: Literal["array"] = "array"
    items: ToolParameter
    
    def to_dict(self):
        return { 
            "items": self.items.to_dict() 
        }
    
class Tool(BaseModel):
    type: str
    name: str

    def to_dict(self):
        return {
            "type": self.type,
            self.type: {
                "name": self.name
            }
        }

class FunctionTool(Tool):
    type: Literal["function"] = "function"
    description: str
    parameters: Union[ToolObjectParameter, ToolArrayParameter, ToolParameter]

    def to_dict(self):
        return {
            "type": self.type,
            self.type: {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_dict()
            }
        }        

class CodeInterpreterTool(Tool):
    type: Literal["code_interpreter"] = "code_interpreter"
    name: str = "code_interpreter"

class FileSearchTool(Tool):
    type: Literal["file_search"] = "file_search"
    name: str = "file_search"

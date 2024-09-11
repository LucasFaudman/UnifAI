from typing import Optional, Mapping, Any
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    tool_name: str
    arguments: Mapping[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    type: str = "function"
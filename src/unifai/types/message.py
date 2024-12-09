from typing import Optional, Literal, Union, Sequence, Any
from datetime import datetime

from ._base_model import BaseModel, Field

from .image import Image
from .tool_call import ToolCall
from .response_info import ResponseInfo, ListWithResponseInfo

class Message(BaseModel):
    # id: str
    role: Literal['user', 'assistant', 'tool', 'system']
    content: Optional[str] = None
    images: Optional[list[Image]] = None
    tool_calls: Optional[list[ToolCall]] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    response_info: Optional[ResponseInfo] = None

    def get_content(self) -> str:
        return self.content or ""


class MessageChunk(Message):
    pass

class Messages(ListWithResponseInfo[Message]):
    pass

class MessageChunks(ListWithResponseInfo[MessageChunk]):
    pass
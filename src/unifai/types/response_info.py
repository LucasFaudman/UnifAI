from typing import Optional, Literal
from pydantic import BaseModel

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens
    
class ResponseInfo(BaseModel):
    model: Optional[str] = None    
    done_reason: Optional[Literal["stop", "tool_calls", "max_tokens", "content_filter"]] = None
    usage: Optional[Usage] = None
    # duration: Optional[int]
    # created_at: datetime = Field(default_factory=datetime.now)    
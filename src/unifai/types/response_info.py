from typing import Optional, Literal, Self
from pydantic import BaseModel

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens
    
    def __iadd__(self, other) -> Self:        
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self
    
    def __add__(self, other) -> "Usage":
        return Usage(input_tokens=self.input_tokens + other.input_tokens, output_tokens=self.output_tokens + other.output_tokens)

    
class ResponseInfo(BaseModel):
    model: Optional[str] = None    
    done_reason: Optional[Literal["stop", "tool_calls", "max_tokens", "content_filter"]] = None
    usage: Optional[Usage] = None
    # duration: Optional[int]
    # created_at: datetime = Field(default_factory=datetime.now)    
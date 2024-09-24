from typing import Optional, Literal, Union, Sequence, Any

from pydantic import BaseModel

from .response_info import ResponseInfo


class Embedding(BaseModel):
    vector: list[float]
    index: int

class EmbedResult(BaseModel):
    embeddings: list[Embedding]
    response_info: Optional[ResponseInfo] = None
    
    def __getitem__(self, index: int) -> Embedding:
        return self.embeddings[index]
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __iter__(self):
        return iter(self.embeddings)
    
    def __contains__(self, item: Embedding) -> bool:
        return item in self.embeddings
    
    def __add__(self, other: "EmbedResult") -> "EmbedResult":
        return EmbedResult(
            embeddings=self.embeddings + other.embeddings, 
            response_info=ResponseInfo(
                model=self.response_info.model or other.response_info.model,
                done_reason=self.response_info.done_reason or other.response_info.done_reason,
                usage=self.response_info.usage + other.response_info.usage if self.response_info.usage and other.response_info.usage else None
            ) if self.response_info and other.response_info else None
        )
    
    def __iadd__(self, other: "EmbedResult") -> "EmbedResult":
        self.embeddings += other.embeddings
        if self.response_info and self.response_info.usage and other.response_info and other.response_info.usage:
            self.response_info.usage += other.response_info.usage
        return self
    
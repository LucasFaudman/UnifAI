from typing import Optional, Literal, Union, Sequence, Any

from pydantic import BaseModel, RootModel, ConfigDict

from .response_info import ResponseInfo

Embedding = list[float]

class Embeddings(RootModel[list[Embedding]]):

    def __init__(self, root: list[list[float]], response_info: Optional[ResponseInfo] = None):
        super().__init__(root=root)
        self.response_info = response_info
    
    def list(self) -> list[list[float]]:
        return self.root     
        
    @property
    def response_info(self) -> Optional[ResponseInfo]:
        if not hasattr(self, "_response_info"):
            self._response_info = None
        return self._response_info

    @response_info.setter
    def response_info(self, response_info: Optional[ResponseInfo]):
        self._response_info = response_info

           
    def __add__(self, other: "Embeddings") -> "Embeddings": 
        return Embeddings(
            root = self.list() + other.list(),
            response_info=ResponseInfo(
                model=self.response_info.model or other.response_info.model,
                done_reason=self.response_info.done_reason or other.response_info.done_reason,
                usage=self.response_info.usage + other.response_info.usage if self.response_info.usage and other.response_info.usage else None
                ) if self.response_info and other.response_info else None
        )
    
    def __iadd__(self, other: "Embeddings") -> "Embeddings":
        self.root += other.list()
        if self.response_info and self.response_info.usage and other.response_info and other.response_info.usage:
            self.response_info.usage += other.response_info.usage
        return self

    def __len__(self) -> int:
        return self.root.__len__()
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Embeddings):
            return False
        return self.root == other.root and self.response_info == other.response_info
    
    def __getitem__(self, index: int) -> Embedding:
        return self.root[index]
    
    def __setitem__(self, index: int, value: Embedding):
        self.root[index] = value

    def __contains__(self, item: Embedding) -> bool:
        return item in self.root
    
    def __iter__(self):
        return self.root.__iter__()
        # return iter(self.root)
    
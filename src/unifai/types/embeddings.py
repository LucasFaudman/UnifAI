from typing import Optional, Literal, Union, Self, Any

from pydantic import BaseModel, RootModel, ConfigDict

from .response_info import ResponseInfo
from unifai.exceptions.embedding_errors import EmbeddingDimensionsError

Embedding = list[float]

# def normalize_l2(x):
#     x = np.array(x)
#     if x.ndim == 1:
#         norm = np.linalg.norm(x)
#         if norm == 0:
#             return x
#         return x / norm
#     else:
#         norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
#         return np.where(norm == 0, x, x / norm)

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


    @property
    def dimensions(self) -> int:
        return len(self.root[0]) if self.root else 0
    

    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        current_dimensions = self.dimensions
        if dimensions < 1 or dimensions > current_dimensions:
            raise EmbeddingDimensionsError(f"Cannot reduce dimensions from {current_dimensions} to {dimensions}. Dimensions cannot be greater than the current dimensions or less than 1.")
        elif dimensions != current_dimensions:
            self.root = [embedding[:dimensions] for embedding in self.root]
        

    def reduce_dimensions(self, dimensions: int) -> Self:
        self.dimensions = dimensions
        return self

           
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
    
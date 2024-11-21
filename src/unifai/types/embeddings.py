from typing import Optional, Literal, Union, Self, Any, List, TypeVar

from pydantic import BaseModel, RootModel, ConfigDict

from .response_info import ListWithResponseInfo, ResponseInfo
from unifai.exceptions.embedding_errors import EmbeddingDimensionsError

T = TypeVar('T')
Embedding = list[float]

class Embeddings(ListWithResponseInfo[Embedding]):
    
    def __init__(
            self, 
            root: List[Embedding], 
            response_info: Optional[ResponseInfo] = None,
            dimensions: Optional[int] = None
        ):
        super().__init__(root=root, response_info=response_info)
        if dimensions:
            self.dimensions = dimensions
    
    @property
    def dimensions(self) -> int:
        return len(self.root[0]) if self.root else 0
    
    @dimensions.setter
    def dimensions(self, dimensions: int) -> None:
        current_dimensions = self.dimensions
        if dimensions < 1 or dimensions > current_dimensions:
            raise EmbeddingDimensionsError(f"Cannot reduce dimensions from {current_dimensions} to {dimensions}. Dimensions cannot be greater than the current dimensions or less than 1.")
        if dimensions != current_dimensions:
            self.root = [embedding[:dimensions] for embedding in self.root]
        
    def reduce_dimensions(self, dimensions: int) -> Self:
        self.dimensions = dimensions
        return self
    

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

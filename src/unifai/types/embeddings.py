from typing import Optional, Literal, Union, Self, Any

from pydantic import BaseModel, RootModel, ConfigDict

# from .response_info import ResponseInfo
from .response_info import ListWithResponseInfo
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


class Embeddings(ListWithResponseInfo[Embedding]):
    
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

# class Embeddings(RootModel[list[Embedding]]):

#     def __init__(self, root: list[list[float]], response_info: Optional[ResponseInfo] = None):
#         super().__init__(root=root)
#         self.response_info = response_info
    

#     def list(self) -> list[list[float]]:
#         return self.root     


#     @property
#     def response_info(self) -> ResponseInfo:
#         if (response_info := getattr(self, "_response_info", None)) is None:
#             response_info = ResponseInfo()
#             self._response_info = response_info
#         return response_info


#     @response_info.setter
#     def response_info(self, response_info: Optional[ResponseInfo]):
#         self._response_info = response_info


#     @property
#     def dimensions(self) -> int:
#         return len(self.root[0]) if self.root else 0
    

#     @dimensions.setter
#     def dimensions(self, dimensions: int) -> None:
#         current_dimensions = self.dimensions
#         if dimensions < 1 or dimensions > current_dimensions:
#             raise EmbeddingDimensionsError(f"Cannot reduce dimensions from {current_dimensions} to {dimensions}. Dimensions cannot be greater than the current dimensions or less than 1.")
#         elif dimensions != current_dimensions:
#             self.root = [embedding[:dimensions] for embedding in self.root]
        

#     def reduce_dimensions(self, dimensions: int) -> Self:
#         self.dimensions = dimensions
#         return self

           
#     def __add__(self, other: "Embeddings") -> "Embeddings": 
#         return Embeddings(
#             root = self.list() + other.list(),
#             response_info=self.response_info + other.response_info
#         )
    

#     def __iadd__(self, other: "Embeddings") -> "Embeddings":
#         self.root += other.list()
#         self.response_info += other.response_info
#         return self


#     def __len__(self) -> int:
#         return self.root.__len__()
    

#     def __eq__(self, other: Any) -> bool:
#         if isinstance(other, Embeddings):
#             return self.root == other.root
#         return self.root == other
    

#     def __getitem__(self, index: int) -> Embedding:
#         return self.root[index]
    

#     def __setitem__(self, index: int, value: Embedding):
#         self.root[index] = value


#     def __contains__(self, item: Embedding) -> bool:
#         return item in self.root
    

#     def __iter__(self):
#         return self.root.__iter__()
    
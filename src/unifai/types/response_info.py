from typing import Optional, Literal, Union, Self, Any, TypeVar, Generic, List, Generator, Type
from pydantic import BaseModel, Field, RootModel

T = TypeVar('T')

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens
    
    def __iadd__(self, other) -> Self:        
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self
    
    def __add__(self, other) -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens, 
            output_tokens=self.output_tokens + other.output_tokens
        )

    
class ResponseInfo(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    done_reason: Optional[Literal["stop", "tool_calls", "max_tokens", "content_filter", "error"]] = None

    def __iadd__(self, other) -> Self:
        self.usage += other.usage
        self.provider = self.provider or other.provider
        self.model = self.model or other.model
        self.done_reason = self.done_reason or other.done_reason
        return self
    
    def __add__(self, other) -> "ResponseInfo":
        return ResponseInfo(
            provider=self.provider or other.provider,
            model=self.model or other.model,
            usage=self.usage + other.usage,
            done_reason=self.done_reason or other.done_reason
        )

class ListWithResponseInfo(RootModel[List[T]], Generic[T]):
    """Generic base class for lists that include response info"""
    
    def __init__(self, root: List[T], response_info: Optional[ResponseInfo] = None):
        super().__init__(root=root)
        self.response_info = response_info

    def list(self) -> List[T]:
        return self.root     

    @property
    def response_info(self) -> ResponseInfo:
        if (response_info := getattr(self, "_response_info", None)) is None:
            response_info = ResponseInfo()
            self._response_info = response_info
        return response_info

    @response_info.setter
    def response_info(self, response_info: Optional[ResponseInfo]):
        self._response_info = response_info
           
    def __add__(self, other: Self) -> Self:
        return self.__class__(
            root = self.list() + other.list(),
            response_info=self.response_info + other.response_info
        )

    def __iadd__(self, other: Self) -> Self:
        self.root += other.list()
        self.response_info += other.response_info
        return self

    def __len__(self) -> int:
        return self.root.__len__()
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return self.root == other.root
        return self.root == other
    
    def __getitem__(self, index: int) -> T:
        return self.root[index]
    
    def __setitem__(self, index: int, value: T):
        self.root[index] = value

    def __contains__(self, item: T) -> bool:
        return item in self.root
    
    def __iter__(self):
        return self.root.__iter__()
    
    def append(self, item: T) -> None:
        """Append a single item to the list"""
        self.root.append(item)

    def extend(self, other: Union[List[T], Self]) -> None:
        """Extend the list with another list or ListWithResponseInfo"""
        if isinstance(other, ListWithResponseInfo):
            self.root.extend(other.list())
            self.response_info += other.response_info
        else:
            self.root.extend(other)    

    @classmethod
    def from_generator(cls: Type[Self], generator: Generator[T, None, ResponseInfo]) -> Self:
        """Create a ListWithResponseInfo from a generator"""
        root = list(generator)
        # response_info = generator.gi_frame.f_locals['response_info']    
        try:
            generator.send(None)  # This will raise StopIteration
        except StopIteration as e:
            response_info = e.value
                    
        instance = cls(root)
        instance.response_info = response_info
        return instance
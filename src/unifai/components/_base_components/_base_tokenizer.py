from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, AbstractSet, Collection
from abc import abstractmethod

from ...exceptions import UnsupportedFeatureError
from ._base_component import UnifAIComponent
from ._base_adapter import UnifAIAdapter
from ...configs.tokenizer_config import TokenizerConfig


T = TypeVar("T")
class Tokenizer(UnifAIComponent[TokenizerConfig]):
    component_type = "tokenizer"
    provider = "tokenizer"
    config_class = TokenizerConfig

    default_tokenizer_model = "gpt2"
    default_encoding = "cl100k_base"
    default_allowed_special: Literal["all"]|AbstractSet[str] = set()
    default_disallowed_special: Literal["all"]|Collection[str] = "all"    

    # Abstract Methods
    @abstractmethod
    def _encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            **kwargs
    ) -> list[int]:
        ...
    
    @abstractmethod
    def _decode(
            self,
            token_ids: list[int],
            model: Optional[str] = None,
            **kwargs
    ) -> str:
        ...
    
    @abstractmethod
    def _tokenize(
            self,
            text: str,
            model: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        ...    
    
    def _count_tokens(
            self, 
            text: str, 
            model: Optional[str] = None,                 
            **kwargs) -> int:
        try:
            return len(self.encode(text, model, **kwargs))
        except (NotImplementedError, UnsupportedFeatureError):
            return len(self.tokenize(text, model, **kwargs))    
    
    def decode(
            self,
            token_ids: list[int],
            model: Optional[str] = None,
            **kwargs
    ) -> str:
        return self._run_func(self._decode, token_ids, model, **kwargs)

    def encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            **kwargs
    ) -> list[int]:
        return self._run_func(self._encode, text, model, **kwargs)
    
    def tokenize(
            self,
            text: str,
            model: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        return self._run_func(self._tokenize, text, model, **kwargs)

    def count_tokens(
            self, 
            text: str, 
            model: Optional[str] = None,                 
            **kwargs) -> int:
        return self._run_func(self._count_tokens, text, model, **kwargs)

    @property
    def default_model(self) -> str:
        return self.config.default_model or self.default_tokenizer_model

class TokenizerAdapter(Tokenizer, UnifAIAdapter):
    """UnifAIAdapter Base Class for Tokenizers"""


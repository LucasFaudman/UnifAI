from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Iterable,  Callable, Iterator, Iterable, Generator, Self

from ...exceptions import UnsupportedFeatureError
from .._base_component import UnifAIComponent
from ..base_adapters._base_adapter import UnifAIAdapter
# from ...types import Document, Documents


T = TypeVar("T")
class Tokenizer(UnifAIComponent):
    provider = "base_tokenizer"
    default_encoding = "cl100k_base"

    def _encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            **kwargs
    ) -> list[int]:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            **kwargs
    ) -> list[int]:
        return self.run_func_convert_exceptions(self._encode, text, model, **kwargs)

    def _decode(
            self,
            token_ids: list[int],
            model: Optional[str] = None,
            **kwargs
    ) -> str:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def decode(
            self,
            token_ids: list[int],
            model: Optional[str] = None,
            **kwargs
    ) -> str:
        return self.run_func_convert_exceptions(self._decode, token_ids, model, **kwargs)

    def _tokenize(
            self,
            text: str,
            model: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def tokenize(
            self,
            text: str,
            model: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        return self.run_func_convert_exceptions(self._tokenize, text, model, **kwargs)

    def _count_tokens(
            self, 
            text: str, 
            model: Optional[str] = None,                 
            **kwargs) -> int:
        try:
            return len(self.encode(text, model, **kwargs))
        except (NotImplementedError, UnsupportedFeatureError):
            return len(self.tokenize(text, model, **kwargs))

    def count_tokens(
            self, 
            text: str, 
            model: Optional[str] = None,                 
            **kwargs) -> int:
        return self.run_func_convert_exceptions(self._count_tokens, text, model, **kwargs)


class TokenizerAdapter(Tokenizer, UnifAIAdapter):
    """UnifAIAdapter Base Class for Tokenizers"""


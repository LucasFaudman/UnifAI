from typing import TYPE_CHECKING, Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, AbstractSet, Collection
from ...exceptions import ProviderUnsupportedFeatureError
from ._base_tokenizer import Tokenizer

T = TypeVar("T")

class StrFuncTestingTokenizer(Tokenizer):
    provider = "str_test_tokenizer"
    default_sep = None

    def __init__(self,
                 sep: Optional[str] = None,
                 support_encode_decode: bool = True,
                 ):
        self.sep = sep or self.default_sep
        self.support_encode_decode = support_encode_decode
        if self.support_encode_decode:
            self._ints_to_tokens: dict[int, str] = {}
            self._tokens_to_ints: dict[str, int] = {}
        
    def _encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            **kwargs
    ) -> list[int]:
        if not self.support_encode_decode:
            raise ProviderUnsupportedFeatureError(f"{self.__class__.__name__} tokenizer does not support encoding. Init with support_encode_decode=True to enable encoding for testing.")
        current_int = len(self._tokens_to_ints)
        _ints = []
        for token in self.tokenize(text):
            if (token_int := self._tokens_to_ints.get(token)) is None:
                self._tokens_to_ints[token] = current_int
                self._ints_to_tokens[current_int] = token
                token_int = current_int
                current_int += 1
            _ints.append(token_int)
        return _ints
            
    def _decode(
            self,
            tokens: list[int],
            model: Optional[str] = None,
            **kwargs
    ) -> str:
        if not self.support_encode_decode:
            raise ProviderUnsupportedFeatureError(f"{self.__class__.__name__} tokenizer does not support decoding. Init with support_encode_decode=True to enable decoding for testing.")                
        return (self.sep or " ").join(self._ints_to_tokens[token] for token in tokens)

class StrLenTokenizer(StrFuncTestingTokenizer):
    provider = "str_len"
    default_sep = ""

    def _count_tokens(self, text: str, model: Optional[str] = None, **kwargs) -> int:
        return len(text)

    def _tokenize(self, text: str, model: Optional[str] = None, **kwargs) -> list[str]:
        return list(text)

class StrSplitTokenizer(StrFuncTestingTokenizer):
    provider = "str_split"

    def _tokenize(self, text: str, model: Optional[str] = None, sep: Optional[str] = None, **kwargs) -> list[str]:
        return text.split(sep or self.sep)


    
from typing import TYPE_CHECKING, Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, AbstractSet, Collection

# if TYPE_CHECKING:
#     import tiktoken
import tiktoken

from ...exceptions import UnifAIError, UnknownUnifAIError, TokenizerVocabError, TokenizerDisallowedSpecialTokenError
from .._base_components._base_component import convert_exceptions
from .._base_components._base_tokenizer import TokenizerAdapter
from ...utils import lazy_import


T = TypeVar("T")

class TikTokenTokenizer(TokenizerAdapter):
    provider = "tiktoken"

    default_encoding = "gpt2"
    default_allowed_special: Literal["all"]|AbstractSet[str] = set()
    default_disallowed_special: Literal["all"]|Collection[str] = "all"

    _encoding_cache: ClassVar[dict[str, tiktoken.Encoding]] = {}

    def import_client(self):
        return lazy_import("tiktoken")

    def init_client(self, **init_kwargs):
        if (allowed_special := init_kwargs.get("allowed_special") or init_kwargs.pop("default_allowed_special", None)) is not None:
            self.default_allowed_special = allowed_special
        if (disallowed_special := init_kwargs.get("disallowed_special") or init_kwargs.pop("default_disallowed_special", None)) is not None:
            self.default_disallowed_special = disallowed_special
        self.init_kwargs.update(init_kwargs)
        self._client = self.import_client()
        return self._client
    
    def convert_exception(self, exception: Exception) -> UnifAIError:
        if isinstance(exception, ValueError) and exception.args and "disallowed special token" in (message := exception.args[0]):
            return TokenizerDisallowedSpecialTokenError(message=message, original_exception=exception)
        if isinstance(exception, KeyError) and exception.args and "Unknown encoding" in (message := exception.args[0]):
            return TokenizerVocabError(message=message, original_exception=exception)
        return UnknownUnifAIError(message=str(exception), original_exception=exception)

    def get_tiktoken_encoding(self, model: Optional[str] = None, encoding: Optional[str] = None) -> tiktoken.Encoding:
        if model:
            encoding = tiktoken.encoding_name_for_model(model)
        if not encoding:
            encoding = self.default_encoding
        if not (tiktoken_encoding := self._encoding_cache.get(encoding)):
            tiktoken_encoding = tiktoken.get_encoding(encoding)
            self._encoding_cache[encoding] = tiktoken_encoding
        return tiktoken_encoding
    
    def _add_default_specials(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "allowed_special" not in kwargs:
            kwargs["allowed_special"] = self.default_allowed_special
        if "disallowed_special" not in kwargs:
            kwargs["disallowed_special"] = self.default_disallowed_special
        return kwargs
    
    def _encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            encoding: Optional[str] = None,
            **kwargs
    ) -> list[int]:
        return self.get_tiktoken_encoding(model, encoding).encode(text, **self._add_default_specials(kwargs))
    
    def _decode(
            self,
            tokens: list[int],
            model: Optional[str] = None,
            encoding: Optional[str] = None,
            **kwargs
    ) -> str:
        return self.get_tiktoken_encoding(model, encoding).decode(tokens, **kwargs)

    def _tokenize(
            self,
            text: str,
            model: Optional[str] = None,
            encoding: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        tiktoken_encoding = self.get_tiktoken_encoding(model, encoding)
        tokens = tiktoken_encoding.encode(text, **self._add_default_specials(kwargs))
        return [tiktoken_encoding.decode_single_token_bytes(token).decode() for token in tokens]
from typing import TYPE_CHECKING, Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, AbstractSet, Collection

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from ...exceptions import UnifAIError, UnknownUnifAIError, TokenizerVocabError, TokenizerDisallowedSpecialTokenError
from .._base_components._base_tokenizer import TokenizerAdapter
from ...utils import lazy_import

T = TypeVar("T")

class HuggingFaceTokenizer(TokenizerAdapter):
    provider = "huggingface"

    default_model = "bert-base-cased"
    default_allowed_special: Literal["all"]|AbstractSet[str] = set()
    default_disallowed_special: Literal["all"]|Collection[str] = "all"

    _cache: ClassVar[dict[str, PreTrainedTokenizerBase]] = {}
    
    def import_client(self):
        return lazy_import("transformers.AutoTokenizer")

    def init_client(self, **client_kwargs):
        self.client_kwargs.update(client_kwargs)
        self._client = self.import_client()
        return self._client

    # def convert_exception(self, exception: Exception) -> UnifAIError:
    #     if isinstance(exception, ValueError) and exception.args and "disallowed special token" in (message := exception.args[0]):
    #         return TokenizerDisallowedSpecialTokenError(message=message, original_exception=exception)
    #     if isinstance(exception, KeyError) and exception.args and "Unknown encoding" in (message := exception.args[0]):
    #         return TokenizerVocabError(message=message, original_exception=exception)
    #     return UnknownUnifAIError(message=str(exception), original_exception=exception)

    def get_autotokenizer(self, model: Optional[str] = None, **kwargs) -> PreTrainedTokenizerBase:
        model = model or self.default_model
        if not (hf_tokenizer := self._cache.get(model)):
            hf_tokenizer = self.client.from_pretrained(model, **{**self.client_kwargs, **kwargs})                
            self._cache[model] = hf_tokenizer
        return hf_tokenizer
        
    def _encode(
            self, 
            text: str, 
            model: Optional[str] = None,                    
            **kwargs
    ) -> list[int]:
        if "add_special_tokens" not in kwargs:
            kwargs["add_special_tokens"] = False           
        return self.get_autotokenizer(model).encode(text, **kwargs)
    
    def _decode(
            self,
            tokens: list[int],
            model: Optional[str] = None,
            **kwargs
    ) -> str:
        if "clean_up_tokenization_spaces" not in kwargs:
            kwargs["clean_up_tokenization_spaces"] = True
        if "skip_special_tokens" not in kwargs:
            kwargs["skip_special_tokens"] = True
        return self.get_autotokenizer(model).decode(tokens, **kwargs)

    def _tokenize(
            self,
            text: str,
            model: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        if "add_special_tokens" not in kwargs:
            kwargs["add_special_tokens"] = False    
        return self.get_autotokenizer(model).tokenize(text, **kwargs)

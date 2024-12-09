import pytest
from typing import Optional, Literal

from unifai import UnifAI
from unifai.components._base_components._base_tokenizer import Tokenizer
# from unifai.exceptions import BadRequestError, NotFoundError, DocumentNotFoundError
from basetest import base_test, base_test_tokenizers_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS

@base_test_tokenizers_all
def test_init_tokenizers(provider, client_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])
    tokenizer = ai.get_tokenizer(provider)
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.provider == provider
    

@base_test_tokenizers_all
def test_tokenize_hello_world(provider, client_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])

    tokenizer = ai.get_tokenizer(provider)
    hello_world = "Hello world"
    token_ids = tokenizer.encode(hello_world)
    print("tokenizer.encode Token Ids:", token_ids)
    assert all(isinstance(token_id, int) for token_id in token_ids)

    decoded = tokenizer.decode(token_ids)
    print("tokenizer.decode Decoded:", decoded)
    assert decoded == hello_world

    tokens = tokenizer.tokenize(hello_world)
    print("tokenizer.tokenize Tokens:", tokens)
    for token in tokens:
        assert isinstance(token, str)
        assert len(token) > 0
        assert token in hello_world #or token.startswith(("[", "Ġ"))

    token_count = tokenizer.count_tokens(hello_world)
    # special_tokens = tokenizer.
    print("tokenizer.count_tokens Token Count:", token_count)
    assert token_count == len(tokens)
    assert token_count == len(token_ids)
    assert len(hello_world.split()) == token_count

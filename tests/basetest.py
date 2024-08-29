import pytest
from simplifai import SimplifAI


def base_test_all_providers(func):
    return pytest.mark.parametrize("provider, client_kwargs, func_kwargs", [
    # ("anthropic", {"api_key": "test"}),
    (
        "openai", 
        {"api_key": "sk-proj-JLo9QCF0Xrwu1s8X66fUT3BlbkFJ8y0422BoUjUgvWD1G6Qo"},
        {}
    ),
    (
        "ollama", 
        {"host": "http://librem-2.local:11434"},
        {"keep_alive": "10m", 'model': 'firefunction-v2'}
    ),
    ])(func)
    # @pytest.mark.parametrize("provider, client_kwargs", [
    # # ("anthropic", {"api_key": "test"}),
    # ("openai", {"api_key": "sk-proj-JLo9QCF0Xrwu1s8X66fUT3BlbkFJ8y0422BoUjUgvWD1G6Qo"}),
    # ("ollama", {"host": "http://librem-2.local:11434"}),
    # ])
    # def wrapper(provider, client_kwargs, *args, **kwargs):
    #     ai = SimplifAI({
    #         provider: client_kwargs
    #     })
    #     return func(ai, provider, client_kwargs, *args, **kwargs)
    # return wrapper


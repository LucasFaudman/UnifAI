import pytest
from unifai import UnifAI
from basetest import base_test_llms_all

@base_test_llms_all
def test_list_models(provider, init_kwargs, func_kwargs):
    ai = UnifAI(provider_configs={provider: init_kwargs})
    for provider_arg in [provider, None]:
        models = ai.list_models(provider_arg)
        assert models
        assert isinstance(models, list)
        assert isinstance(models[0], str)
        print(f'{provider} Models: \n' + '\n'.join(models))


import pytest
from unifai import UnifAI
from basetest import base_test_llms, API_KEYS

@base_test_llms
def test_list_models(provider, init_kwargs):
    ai = UnifAI(api_keys=API_KEYS)
    for provider_arg in [provider, None]:
        models = ai.list_llm_models(provider_arg)
        assert models
        assert isinstance(models, list)
        assert isinstance(models[0], str)
        print(f'{provider} Models: \n' + '\n'.join(models))


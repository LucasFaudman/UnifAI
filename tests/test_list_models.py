import pytest
from simplifai import SimplifAIClient
from basetest import base_test_all_providers

@base_test_all_providers
def test_list_models(provider, client_kwargs, func_kwargs):
    ai = SimplifAIClient({provider: client_kwargs})
    for provider_arg in [provider, None]:
        models = ai.list_models(provider_arg)
        assert models
        assert isinstance(models, list)
        assert isinstance(models[0], str)
        print(f'{provider} Models: \n' + '\n'.join(models))


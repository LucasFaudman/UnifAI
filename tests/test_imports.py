import pytest
from unifai import UnifAIClient

def test_import_simplifai_client():
    assert UnifAIClient

@pytest.mark.parametrize("provider, client_kwargs", [
    ("anthropic", {"api_key": "test"}),
    ("openai", {"api_key": "test"}),
    ("ollama", {}),
])
def test_init_clients(provider, client_kwargs):
    ai = UnifAIClient({
        provider: client_kwargs
    })

    assert ai.provider_client_kwargs[provider] == client_kwargs
    assert ai.providers == [provider]
    assert ai._clients == {}
    assert ai.default_provider == provider

    wrapper_name = provider.capitalize().replace("ai", "AI") + "Wrapper"
    # assert wrapper_name not in globals()

    client = ai.init_client(provider)    
    # assert wrapper_name in globals()
    # wrapper = globals()[wrapper_name]

    assert client
    # assert isinstance(client, wrapper)    
    assert ai._clients[provider] == client
    assert ai.get_client(provider) == client
    assert ai.get_client() == client
    
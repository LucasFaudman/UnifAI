import pytest
from unifai import UnifAIClient

def test_import_simplifai_client():
    assert UnifAIClient

@pytest.mark.parametrize("provider, client_kwargs", [
    ("anthropic", {"api_key": "test"}),
    ("openai", {"api_key": "test"}),
    ("ollama", {}),
])
def test_init_ai_clients(provider, client_kwargs):
    ai = UnifAIClient({
        provider: client_kwargs
    })

    assert ai.provider_client_kwargs[provider] == client_kwargs
    assert ai.providers == [provider]
    assert ai._clients == {}
    assert ai.default_ai_provider == provider

    wrapper_name = provider.capitalize().replace("ai", "AI") + "Wrapper"
    # assert wrapper_name not in globals()

    client = ai.init_component(provider)    
    # assert wrapper_name in globals()
    # wrapper = globals()[wrapper_name]

    assert client
    # assert isinstance(client, wrapper)    
    assert ai._clients[provider] is client
    assert ai.get_component(provider) is client
    assert ai.get_ai_client() is client
    assert ai.get_ai_client(provider) is client

    

@pytest.mark.parametrize("provider, client_kwargs", [
    ("chroma", {"api_key": "test"}),
    ("pinecone", {"api_key": "test"}),
])
def test_init_vector_db_clients(provider, client_kwargs):
    ai = UnifAIClient({
        provider: client_kwargs
    })

    assert ai.provider_client_kwargs[provider] == client_kwargs
    assert ai.providers == [provider]
    assert ai._clients == {}
    assert ai.default_vector_db_provider == provider

    client = ai.init_component(provider)    

    assert client
    assert ai._clients[provider] is client
    assert ai.get_component(provider) is client
    assert ai.get_vector_db() is client 
    assert ai.get_vector_db(provider) is client   
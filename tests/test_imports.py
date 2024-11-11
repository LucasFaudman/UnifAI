import pytest
from unifai import UnifAI

def test_import_simplifai_client():
    assert UnifAI

@pytest.mark.parametrize("provider, client_kwargs", [
    ("anthropic", {"api_key": "test"}),
    ("openai", {"api_key": "test"}),
    ("ollama", {}),
])
def test_init_ai_components(provider, client_kwargs):
    ai = UnifAI({
        provider: client_kwargs
    })

    assert ai.provider_client_kwargs[provider] == client_kwargs
    assert ai.providers == [provider]
    assert ai._components == {}
    assert ai.default_llm_provider == provider



    client = ai.init_component(provider, "llm", **client_kwargs)    
    # assert wrapper_name in globals()
    # wrapper = globals()[wrapper_name]

    assert client
    # assert isinstance(client, wrapper)    
    assert ai._components["llm"][provider] is client
    assert ai.get_component(provider) is client
    assert ai.get_llm_client() is client
    assert ai.get_llm_client(provider) is client

    

@pytest.mark.parametrize("provider, client_kwargs", [
    ("chroma", {"api_key": "test"}),
    ("pinecone", {"api_key": "test"}),
])
def test_init_vector_db_components(provider, client_kwargs):
    ai = UnifAI({
        provider: client_kwargs
    })

    assert ai.provider_client_kwargs[provider] == client_kwargs
    assert ai.providers == [provider]
    assert ai._components == {}
    assert ai.default_vector_db_provider == provider

    client = ai.init_component(provider, "vector_db")

    assert client
    assert ai._components["vector_db"][provider] is client
    assert ai.get_component(provider, "vector_db") is client
    assert ai.get_vector_db() is client 
    assert ai.get_vector_db(provider) is client   
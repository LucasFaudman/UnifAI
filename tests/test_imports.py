import pytest
from unifai import UnifAI

def test_import_simplifai_client():
    assert UnifAI

@pytest.mark.parametrize("provider, init_kwargs", [
    ("anthropic", {"api_key": "test"}),
    ("openai", {"api_key": "test"}),
    ("ollama", {}),
])
def test_init_ai_components(provider, init_kwargs):
    ai = UnifAI(provider_configs={
        provider: init_kwargs
    })

    assert ai.provider_init_kwargs[provider] == init_kwargs
    assert ai.providers == [provider]
    assert ai._components == {}
    assert ai.default_llm_provider == provider



    client = ai._init_component(provider, "llm", **init_kwargs)    
    # assert wrapper_name in globals()
    # wrapper = globals()[wrapper_name]

    assert client
    # assert isinstance(client, wrapper)    
    assert ai._components["llm"][provider] is client
    assert ai._get_component(provider) is client
    assert ai._get_llm() is client
    assert ai._get_llm(provider) is client

    

@pytest.mark.parametrize("provider, init_kwargs", [
    ("chroma", {"api_key": "test"}),
    ("pinecone", {"api_key": "test"}),
])
def test_init_vector_db_components(provider, init_kwargs):
    ai = UnifAI(provider_configs={
        provider: init_kwargs
    })

    assert ai.provider_init_kwargs[provider] == init_kwargs
    assert ai.providers == [provider]
    assert ai._components == {}
    assert ai.default_vector_db_provider == provider

    client = ai._init_component(provider, "vector_db")

    assert client
    assert ai._components["vector_db"][provider] is client
    assert ai._get_component(provider, "vector_db") is client
    assert ai.vector_db() is client 
    assert ai.vector_db(provider) is client   
import pytest
from unifai import UnifAIClient, AIProvider
from unifai.types import Message, Tool, EmbedResult, Embedding, ResponseInfo, Usage
from unifai.exceptions import ProviderUnsupportedFeatureError, BadRequestError
from basetest import base_test_all_providers, base_test_no_anthropic

@pytest.mark.parametrize("input", [
    "Embed this",
    ["Embed this"],
    ["Embed this", "And this"],
    ("Embed this", "And this"),
])
@base_test_all_providers
def test_embeddings_simple(
    provider: AIProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    input: str|list[str]
    ):

    ai = UnifAIClient({provider: client_kwargs})
    ai.init_client(provider, **client_kwargs)

    if provider == "anthropic":
        with pytest.raises(ProviderUnsupportedFeatureError):
            result = ai.embed(input, provider=provider, **func_kwargs)
        return
    
    result = ai.embed(input, provider=provider, **func_kwargs)

    assert isinstance(result, EmbedResult)
    assert isinstance(result.embeddings, list)
    assert all(isinstance(embedding, Embedding) for embedding in result.embeddings)
    assert all(isinstance(embedding.vector, list) for embedding in result.embeddings)
    assert all(isinstance(embedding.vector[0], float) for embedding in result.embeddings)
    assert isinstance(result.response_info, ResponseInfo)
    assert isinstance(result.response_info.usage, Usage)
    assert isinstance(result.response_info.usage.total_tokens, int)
    # assert result.response_info.usage.input_tokens > 0
    # assert result.response_info.usage.output_tokens == 0
    assert result.response_info.usage.total_tokens == result.response_info.usage.input_tokens


    assert result[0] == result.embeddings[0]
    assert isinstance(result[0], Embedding)
    assert len(result) == len(result.embeddings)    
    expected_length = 1 if isinstance(input, str) else len(input) if hasattr(input, "__len__") else 2
    assert len(result) == expected_length

    for i, embedding in enumerate(result):
        assert embedding in result
        assert result[i] == embedding
        assert result[i].vector == embedding.vector
        assert result[i].index == embedding.index
        assert embedding.index == i
        
    other_result = EmbedResult(
        embeddings=[Embedding(vector=[0.0], index=1)], 
        response_info=ResponseInfo(model="other_model", usage=Usage(input_tokens=1, output_tokens=0))
    )
    combined_result = result + other_result
    assert isinstance(combined_result, EmbedResult)
    assert len(combined_result) == len(result) + len(other_result)
    assert combined_result.embeddings == result.embeddings + other_result.embeddings

    result += other_result
    assert isinstance(result, EmbedResult)
    assert len(result) == len(combined_result)
    assert result.embeddings == combined_result.embeddings
    assert result.response_info.model == combined_result.response_info.model


    texts = input if isinstance(input, list) else [input]
    for text, embedding in zip(texts, result):
        print(f"Text: {text}\nEmbedding: {embedding.vector[0]} and {len(embedding.vector) -1 } more\n")

@pytest.mark.parametrize("input, max_dimensions", [
    ("Embed this", 100),
    (["Embed this longer text"], 100),
    ("Embed this", 1000),
    (["Embed this longer text"], 1000),
    ("Embed this", 1),
    (["Embed this longer text"], 1),
])
@base_test_no_anthropic
def test_embeddings_max_dimensions(
    provider: AIProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    input: str|list[str],
    max_dimensions: int
    ):

    ai = UnifAIClient({provider: client_kwargs})
    ai.init_client(provider, **client_kwargs)

    result = ai.embed(input, 
                      provider=provider, 
                      max_dimensions=max_dimensions,
                      **func_kwargs)     
    
    assert isinstance(result, EmbedResult)
    for embedding in result:
        assert len(embedding.vector) <= max_dimensions
        assert all(isinstance(value, float) for value in embedding.vector)


@pytest.mark.parametrize("input, max_dimensions", [
    ("Embed this zero", 0),
    ("Embed this negative", -1),
    ("Embed this huge", 1000000),
])
@base_test_no_anthropic
def test_embeddings_max_dimensions_errors(
    provider: AIProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    input: str|list[str],
    max_dimensions: int
    ):

    ai = UnifAIClient({provider: client_kwargs})
    if max_dimensions > 1 and (provider == "ollama" or provider == "google"):
        result = ai.embed(input, 
                      provider=provider, 
                      max_dimensions=max_dimensions,
                      **func_kwargs)            
        assert isinstance(result, EmbedResult)
        for embedding in result:
            assert len(embedding.vector) <= max_dimensions
            assert all(isinstance(value, float) for value in embedding.vector)
            print(f"Embedding: {embedding.vector[0]} and {len(embedding.vector) -1 } more\n")

    else:                
        with pytest.raises((BadRequestError, ValueError)):
            result = ai.embed(input, 
                      provider=provider, 
                      max_dimensions=max_dimensions,
                      **func_kwargs)
    

import pytest
from unifai import UnifAIClient, LLMProvider
from unifai.types import Message, Tool, Embeddings, Embedding, ResponseInfo, Usage
from unifai.exceptions import ProviderUnsupportedFeatureError, BadRequestError
from basetest import base_test_all_llms, base_test_no_anthropic

@pytest.mark.parametrize("input", [
    "Embed this",
    ["Embed this"],
    ["Embed this", "And this"],
    ("Embed this", "And this"),
])
@base_test_all_llms
def test_embeddings_simple(
    provider: LLMProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    input: str|list[str]
    ):

    ai = UnifAIClient({provider: client_kwargs})
    ai.init_client(provider, **client_kwargs)

    if provider == "anthropic":
        with pytest.raises((ProviderUnsupportedFeatureError, AttributeError)):
            result = ai.embed(input, provider=provider, **func_kwargs)
        return
    
    result = ai.embed(input, provider=provider, **func_kwargs)

    assert isinstance(result, Embeddings)
    assert isinstance(result.list(), list)
    assert all(isinstance(embedding, list) for embedding in result)
    assert all(isinstance(embedding[0], float) for embedding in result)
    assert isinstance(result.response_info, ResponseInfo)
    assert isinstance(result.response_info.usage, Usage)
    assert isinstance(result.response_info.usage.total_tokens, int)
    # assert result.response_info.usage.input_tokens > 0
    # assert result.response_info.usage.output_tokens == 0
    assert result.response_info.usage.total_tokens == result.response_info.usage.input_tokens


    assert result[0] == result[0]
    assert isinstance(result[0], list)
    assert len(result) == len(result)    
    expected_length = 1 if isinstance(input, str) else len(input) if hasattr(input, "__len__") else 2
    assert len(result) == expected_length

    for i, embedding in enumerate(result):
        assert embedding in result
        assert result[i] == embedding
        assert result[i] == embedding

        
    other_result = Embeddings(
        root=[[0.1]], 
        response_info=ResponseInfo(model="other_model", usage=Usage(input_tokens=1, output_tokens=0))
    )
    combined_result = result + other_result
    assert isinstance(combined_result, Embeddings)
    assert len(combined_result) == len(result) + len(other_result)
    assert combined_result == result + other_result

    result += other_result
    assert isinstance(result, Embeddings)
    assert len(result) == len(combined_result)
    assert result == combined_result
    assert result.response_info and combined_result.response_info
    assert result.response_info.model == combined_result.response_info.model

    texts = input if isinstance(input, list) else [input]
    for text, embedding in zip(texts, result):
        print(f"Text: {text}\nEmbedding: {embedding[0]} and {len(embedding) -1 } more\n")

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
    provider: LLMProvider, 
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
    
    assert isinstance(result, Embeddings)
    for embedding in result:
        assert len(embedding) <= max_dimensions
        assert all(isinstance(value, float) for value in embedding)


@pytest.mark.parametrize("input, max_dimensions", [
    ("Embed this zero", 0),
    ("Embed this negative", -1),
    ("Embed this huge", 1000000),
])
@base_test_no_anthropic
def test_embeddings_max_dimensions_errors(
    provider: LLMProvider, 
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
        assert isinstance(result, Embeddings)
        for embedding in result:
            assert len(embedding) <= max_dimensions
            assert all(isinstance(value, float) for value in embedding)
            print(f"Embedding: {embedding[0]} and {len(embedding) -1 } more\n")

    else:                
        with pytest.raises((BadRequestError, ValueError)):
            result = ai.embed(input, 
                      provider=provider, 
                      max_dimensions=max_dimensions,
                      **func_kwargs)
    

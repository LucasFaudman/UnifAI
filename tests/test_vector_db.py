import pytest
from typing import Optional, Literal

from unifai import UnifAIClient, AIProvider, VectorDBProvider, Provider
from unifai.wrappers.vector_db_clients import BaseVectorDBClient, BaseVectorDBIndex, ChromaClient, ChromaIndex
from unifai.types import VectorDBProvider, VectorDBGetResult, VectorDBQueryResult, Embedding, Embeddings, ResponseInfo
from unifai.exceptions import BadRequestError
from basetest import base_test_all_db_providers, base_test_db_no_pinecone, PROVIDER_DEFAULTS
from chromadb.errors import InvalidCollectionException

@base_test_db_no_pinecone
def test_init_vector_db_init_clients(provider, client_kwargs, func_kwargs):
    ai = UnifAIClient({
        provider: client_kwargs
    })

    assert ai.provider_client_kwargs[provider] == client_kwargs
    assert ai.providers == [provider]
    assert ai._clients == {}
    assert ai.default_vector_db_provider == provider

    client = ai.init_client(provider)    

    assert client
    assert ai._clients[provider] is client
    assert ai.get_client(provider) is client
    assert ai.get_vector_db_client() is client 
    assert ai.get_vector_db_client(provider) is client 



def parameterize_name_and_metadata(func):
    return pytest.mark.parametrize(
        "name, metadata",
        [
            ("test_index", {"test": "metadata"}),
            # ("test_index", {"test": "metadata", "another": "metadata"}),
        ]
    )(func)

def parameterize_embedding_provider_embedding_model(func):
    return pytest.mark.parametrize(
        "embedding_provider, embedding_model",
        [
            # ("openai", None),
            # ("openai", "text-embedding-3-large"),
            ("openai", "text-embedding-3-small"),
            # ("openai", "text-embedding-ada-002"),
            ("google", None),
            # ("google", "models/text-embedding-004"),
            # ("google", "embedding-gecko-001"),
            # ("google", "embedding-001"),
            ("ollama", None),
            # ("ollama", "llama3.1-8b-num_ctx-8192:latest"),
            # ("ollama", "mistral:latest"),
        ]
    )(func)


def parameterize_dimensions(func):
    return pytest.mark.parametrize(
        "dimensions",
        [
            None, 
            # 100, 
            # 1000, 
            1536, 
            # 3072
        ]
    )(func)

def parameterize_distance_metric(func):
    return pytest.mark.parametrize(
        "distance_metric",
        [
            None, 
            # "cosine", 
            # "euclidean", 
            # "dotproduct"
        ]
    )(func)









@base_test_db_no_pinecone
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_create_index(provider: Provider, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[AIProvider],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]]                                                                
                                ):

    ai = UnifAIClient({
        provider: client_kwargs,
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],        
    })

    client = ai.get_client(provider)
    assert client
    assert isinstance(client, BaseVectorDBClient)
    # if provider == "chroma":
    #     assert isinstance(client, ChromaClient)

    index = client.create_index(
        name=name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric
    )
    assert index
    assert isinstance(index, BaseVectorDBIndex)
    assert index.name == name
    
    updated_metadata = {
                **metadata,
                "_unifai_embedding_config": ",".join((
                str(embedding_provider),
                str(embedding_model),
                str(dimensions),
                str(distance_metric)
            ))                
    }
    assert index.metadata == updated_metadata
    assert index.embedding_provider == embedding_provider
    assert index.embedding_model == embedding_model
    assert index.dimensions == dimensions
    assert index.distance_metric == distance_metric

    assert client.get_index(name) is index

    # assert client.get_indexes() == [index]
    assert client.list_indexes() == [name]
    assert client.count_indexes() == 1

    index2_name = "index_2"
    with pytest.raises(BadRequestError):
        index2 = client.get_index(index2_name)

    index2 = client.get_or_create_index(
        name=index2_name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric
    )

    assert index2
    assert isinstance(index2, BaseVectorDBIndex)
    assert index2.name == index2_name
    assert index2.metadata == updated_metadata
    assert index2.embedding_provider == embedding_provider
    assert index2.embedding_model == embedding_model
    assert index2.dimensions == dimensions
    assert index2.distance_metric == distance_metric

    assert client.get_index(index2_name) is index2
    # assert client.list_indexes() == [index2_name, name]
    assert sorted(client.list_indexes()) == sorted([name, index2_name])
    assert client.count_indexes() == 2
    
    # test getting index by metadata
    client.indexes.pop(index2_name)
    metaloaded_index2 = client.get_index(name=index2_name)
    assert metaloaded_index2
    assert isinstance(metaloaded_index2, BaseVectorDBIndex)
    assert metaloaded_index2.name == index2.name
    assert metaloaded_index2.metadata == index2.metadata
    assert metaloaded_index2.embedding_provider == index2.embedding_provider
    assert metaloaded_index2.embedding_model == index2.embedding_model
    assert metaloaded_index2.dimensions == index2.dimensions
    assert metaloaded_index2.distance_metric == index2.distance_metric

    assert client.get_index(index2_name) == metaloaded_index2

    # test deleting index
    client.delete_index(index2_name)
    assert client.list_indexes() == [name]
    assert client.count_indexes() == 1
    client.delete_index(name)
    assert client.list_indexes() == []
    assert client.count_indexes() == 0
    

def approx_embeddings(embeddings, expected_embeddings):
    assert len(embeddings) == len(expected_embeddings)
    for i, embedding in enumerate(embeddings):
        for j, value in enumerate(embedding):
            assert pytest.approx(value) == pytest.approx(expected_embeddings[i][j])

@base_test_db_no_pinecone
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_add(provider: Provider, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[AIProvider],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]]                                                                
                                ):

    ai = UnifAIClient({
        provider: client_kwargs,
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],        
    })

    client = ai.get_client(provider)
    assert client
    assert isinstance(client, BaseVectorDBClient)
    # if provider == "chroma":
    #     assert isinstance(client, ChromaClient)

    index = client.create_index(
        name=name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric
    )
    assert index
    assert isinstance(index, BaseVectorDBIndex)

    index.add(
        ids=["test_id"],
        metadatas=[{"test": "metadata"}],
        documents=["test document"],
        # embeddings=[Embedding(vector = ([1.0] * dimensions), index=0)]
    )

    # test including embeddings
    assert index.count() == 1
    get_result = index.get(["test_id"])
    assert get_result
    assert get_result.ids == ["test_id"]
    assert get_result.metadatas == [{"test": "metadata"}]
    assert get_result.documents == ["test document"]
    assert get_result.embeddings == None

    get_result = index.get(["test_id"], include=["embeddings"])
    assert get_result
    assert get_result.ids == ["test_id"]
    assert get_result.metadatas == None
    assert get_result.documents == None
    assert get_result.embeddings
    assert len(get_result.embeddings) == 1

    computed_embedding = get_result.embeddings[0]

    if dimensions is None:
        dimensions = len(computed_embedding)

    manual_embeddings = [[.1] * dimensions]
    manual_embeddings2 = [[.2] * dimensions]
    

    index.add(
        ids=["test_id_2"],
        metadatas=[{"test": "metadata2"}],
        documents=["test document2"],
        embeddings=manual_embeddings
    )

    assert index.count() == 2
    get_result = index.get(where={"test": "metadata2"}, include=["metadatas", "documents", "embeddings"])

    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2"}]
    assert get_result.documents == ["test document2"]
    approx_embeddings(get_result.embeddings, manual_embeddings)

    get_result = index.get(["test_id", "test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id", "test_id_2"]
    assert get_result.metadatas == [{"test": "metadata"}, {"test": "metadata2"}]
    assert get_result.documents == ["test document", "test document2"]
    approx_embeddings(get_result.embeddings, [computed_embedding] + manual_embeddings)

    # test updating
    index.update(
        ids=["test_id_2"],
        metadatas=[{"test": "metadata2-UPDATED"}],
        documents=["test document2-UPDATED"],
        embeddings=manual_embeddings2
    )

    assert index.count() == 2
    get_result = index.get(["test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2-UPDATED"}]
    assert get_result.documents == ["test document2-UPDATED"]
    approx_embeddings(get_result.embeddings, manual_embeddings2)

    # test deleting
    index.delete(["test_id_2"])
    assert index.count() == 1
    get_result = index.get(["test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert not get_result.metadatas
    assert not get_result.documents
    assert not get_result.embeddings        

    # test upsert
    index.upsert(
        ids=["test_id_2"],
        metadatas=[{"test": "metadata2-UPDATED"}],
        documents=["test document2-UPDATED"],
        embeddings=manual_embeddings2
    )

    assert index.count() == 2
    get_result = index.get(["test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2-UPDATED"}]


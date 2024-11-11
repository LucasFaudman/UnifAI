import pytest
from typing import Optional, Literal

from unifai import UnifAI, LLMProvider, VectorDBProvider, Provider
from unifai.components.retrievers._base_vector_db_client import VectorDBClient, VectorDBIndex, DocumentDB
from unifai.components import DictDocumentDB

from unifai.types import VectorDBProvider, VectorDBGetResult, VectorDBQueryResult, Embedding, Embeddings, ResponseInfo
from unifai.exceptions import BadRequestError, NotFoundError
from basetest import base_test, base_test_vector_dbs_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS
from chromadb.errors import InvalidCollectionException

from time import sleep
@base_test_vector_dbs_all
def test_init_vector_db_init_clients(provider, client_kwargs, func_kwargs):
    ai = UnifAI({
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



def parameterize_name_and_metadata(func):
    return pytest.mark.parametrize(
        "name, metadata",
        [
            ("test-index", {"test": "metadata"}),
            # ("test-index", {"test": "metadata", "another": "metadata"}),
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
            ("nvidia", None),
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
            1000, 
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









@base_test(*VECTOR_DB_PROVIDERS, exclude=["chroma"])
# @base_test_vector_dbs_all
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_create_index(provider: Provider, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[LLMProvider],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]],                                                                                               
                                tmp_path,
                                serial
                                ):
    # if provider == "chroma":
    #     client_kwargs["persist_directory"] = str(tmp_path)
    # name = f"{name}_{provider}_{embedding_provider}_{embedding_model}_{dimensions}_{distance_metric}"

    ai = UnifAI({
        provider: client_kwargs,
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],        
    })

    client = ai.get_component(provider)
    assert client
    assert isinstance(client, VectorDBClient)
    client.delete_all_indexes() # Reset for each test

    index = client.create_index(
        name=name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        **func_kwargs
    )
    assert index
    assert isinstance(index, VectorDBIndex)
    assert index.name == name
    
    updated_metadata = {
                **metadata,
                "_unifai_embedding_config": ",".join((
                str(embedding_provider),
                str(embedding_model),
                str(dimensions if dimensions is not None else 1536),
                str(distance_metric if distance_metric is not None else "cosine")
            ))                
    }
    if provider == "chroma": assert index.metadata == updated_metadata
    
    assert index.embedding_provider == embedding_provider
    assert index.embedding_model == embedding_model
    assert index.dimensions == dimensions if dimensions is not None else 1536
    assert index.distance_metric == distance_metric if distance_metric is not None else "cosine"

    assert client.get_index(name) is index

    # assert client.get_indexes() == [index]
    assert client.list_indexes() == [name]
    assert client.count_indexes() == 1

    index2_name = "index-2"
    # TODO both should raise InvalidIndexException
    with pytest.raises((BadRequestError, NotFoundError)):
        index2 = client.get_index(index2_name)

    index2 = client.get_or_create_index(
        name=index2_name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        **func_kwargs
    )

    assert index2
    assert isinstance(index2, VectorDBIndex)
    assert index2.name == index2_name
    if provider == "chroma": assert index2.metadata == updated_metadata
    assert index2.embedding_provider == embedding_provider
    assert index2.embedding_model == embedding_model
    assert index2.dimensions == dimensions if dimensions is not None else 1536
    assert index2.distance_metric == distance_metric if distance_metric is not None else "cosine"

    assert client.get_index(index2_name) is index2
    # assert client.list_indexes() == [index2_name, name]
    assert sorted(client.list_indexes()) == sorted([name, index2_name])
    assert client.count_indexes() == 2
    
    # test getting index by metadata
    if provider == "chroma":
        client.indexes.pop(index2_name)
        metaloaded_index2 = client.get_index(name=index2_name)
        assert metaloaded_index2
        assert isinstance(metaloaded_index2, VectorDBIndex)
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
    
    del ai
    

def approx_embeddings(embeddings, expected_embeddings):
    assert len(embeddings) == len(expected_embeddings)
    for i, embedding in enumerate(embeddings):
        for j, value in enumerate(embedding):
            assert pytest.approx(value) == pytest.approx(expected_embeddings[i][j])

# @base_test(*VECTOR_DB_PROVIDERS, exclude=["chroma"])
# @base_test_vector_dbs_all
@base_test(*VECTOR_DB_PROVIDERS, exclude=["chroma"])
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_add(provider: Provider, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[LLMProvider],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]],                                                                
                                # serial
                                ):

    ai = UnifAI({
        provider: client_kwargs,
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],        
    })

    client = ai.get_component(provider)
    assert client
    assert isinstance(client, VectorDBClient)
    client.delete_all_indexes() # Reset for each test

    if provider == "chroma":
        document_db = None
    else:
        document_db = DictDocumentDB({})

    index = client.create_index(
        name=name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        document_db=document_db,
        **func_kwargs
    )
    assert index
    assert isinstance(index, VectorDBIndex)

    index.add(
        ids=["test_id"],
        metadatas=[{"test": "metadata"}],
        documents=["test document"],
        # embeddings=[Embedding(vector = ([1.0] * dimensions), index=0)]
    )

    # test including embeddings
    if provider == 'pinecone': sleep(10)
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

    if provider == 'pinecone': sleep(10)
    assert index.count() == 2
    get_result = index.get(["test_id_2"], where={"test": "metadata2"}, include=["metadatas", "documents", "embeddings"])
    # get_result = index.get(where={"test": "metadata2"}, include=["metadatas", "documents", "embeddings"])

    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2"}]
    assert get_result.documents == ["test document2"]
    approx_embeddings(get_result.embeddings, manual_embeddings)

    get_result = index.get(["test_id", "test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert get_result
    assert get_result.sort().ids == ["test_id", "test_id_2"]
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

    if provider == 'pinecone': sleep(10)
    assert index.count() == 2
    get_result = index.get(["test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2-UPDATED"}]
    assert get_result.documents == ["test document2-UPDATED"]
    approx_embeddings(get_result.embeddings, manual_embeddings2)

    # test deleting
    index.delete(["test_id_2"])
    if provider == 'pinecone': sleep(10)
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

    if provider == 'pinecone': sleep(10)
    assert index.count() == 2
    get_result = index.get(["test_id_2"], include=["metadatas", "documents", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2-UPDATED"}]
    assert get_result.documents == ["test document2-UPDATED"]
    approx_embeddings(get_result.embeddings, manual_embeddings2)

    # Test get/delete all ids
    all_ids = index.list_ids()
    assert all_ids == ["test_id", "test_id_2"]
    index.delete_all()
    if provider == 'pinecone': sleep(10)
    assert index.count() == 0

    # test upsert with multiple
    num_docs = 69
    many_ids, many_metadatas, many_documents, many_embeddings = [], [], [], []
    ids = [(i, f"test_id_{i}") for i in range(num_docs)]
    for i, id in sorted(ids, key=lambda x: x[1]):
        many_ids.append(id)
        many_metadatas.append({"test": f"metadata_{i}"})
        many_documents.append(f"test document_{i}")
        many_embeddings.append([.1] * dimensions)    
        
    index.add(
        ids=many_ids,
        metadatas=many_metadatas,
        documents=many_documents,
        embeddings=many_embeddings
    )
    if provider == 'pinecone': sleep(40)
    assert index.count() == num_docs
    if provider == 'pinecone': sleep(40)
    get_result = index.get(many_ids, include=["metadatas", "documents", "embeddings"])
    assert get_result
    print(get_result.ids)
    assert get_result.sort().ids == many_ids
    assert get_result.ids == many_ids
    assert get_result.metadatas == many_metadatas
    assert get_result.documents == many_documents
    approx_embeddings(get_result.embeddings, many_embeddings)

    # test deleting all
    index.delete_all()
    if provider == 'pinecone': sleep(10)
    assert index.count() == 0

    # test upsert with multiple after deleting all
    index.upsert(
        ids=many_ids,
        metadatas=many_metadatas,
        documents=many_documents,
        embeddings=many_embeddings
    )
    if provider == 'pinecone': sleep(40)
    assert index.count() == num_docs
    if provider == 'pinecone': sleep(40)
    get_result = index.get(many_ids, include=["metadatas", "documents", "embeddings"])
    assert get_result
    print(get_result.ids)
    assert get_result.sort().ids == many_ids
    assert get_result.ids == many_ids
    assert get_result.metadatas == many_metadatas
    assert get_result.documents == many_documents
    approx_embeddings(get_result.embeddings, many_embeddings) 

    del ai

# @base_test_vector_dbs_all
# @base_test(*VECTOR_DB_PROVIDERS, exclude=["pinecone"])

@base_test(*VECTOR_DB_PROVIDERS, exclude=["chroma"])
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_query_simple(provider: Provider, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[LLMProvider],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]],                                                                
                                # serial
                                ):

    ai = UnifAI({
        provider: client_kwargs,
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],        
    })

    client = ai.get_component(provider)
    assert client
    assert isinstance(client, VectorDBClient)
    client.delete_all_indexes() # Reset for each test


    if provider == "chroma":
        document_db = None
    else:
        document_db = DictDocumentDB({})

    index = client.create_index(
        name=name,
        metadata=metadata,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        document_db=document_db,
        **func_kwargs
    )
    assert index
    assert isinstance(index, VectorDBIndex)

    groups = {
        "animals": {
            "all": ["dog", "fish", "cat", "bird", "elephant", "giraffe", "lion", "tiger", "bear", "wolf"],
            "dog": ["poodle", "labrador", "bulldog", "beagle", "dalmatian", "german shepherd", "golden retriever", "husky", "rottweiler", "doberman"],
            "fish": ["goldfish", "bass", "salmon", "trout", "catfish", "perch", "pike", "mackerel", "cod", "haddock"],
        },
        "vehicles": {
            "all": ["car", "truck", "bus", "bike", "motorcycle", "scooter", "skateboard", "rollerblade", "train", "plane"],
            "car": ["toyota", "honda", "ford", "chevrolet", "dodge", "bmw", "audi", "mercedes", "volkswagen", "porsche"],
            "truck": ["semi", "pickup", "dump", "garbage", "tow", "box", "flatbed", "tanker", "fire", "ice cream"],
        },
        "domains": {
            "all": ["city", "ocean", "country", "continent", "sea", "river", "lake", "mountain", "valley", "desert"],
            "city": ["new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio", "san diego", "dallas", "san jose"],
            "ocean": ["pacific", "atlantic", "indian", "arctic", "antarctic", "caribbean", "mediterranean", "south china", "baltic", "gulf of mexico"],
        }
    }

    num_groups = len(groups)
    sub_group_size = 10

    ids, metadatas, documents = [], [], []
    for group_name, group in groups.items():
        for sub_group_name, sub_group in group.items():
            assert len(sub_group) == sub_group_size
            for doc in sub_group:
                ids.append(f"{group_name}_{sub_group_name}_{doc}")
                metadatas.append({"group": group_name, "sub_group": sub_group_name})
                documents.append(doc)

    assert len(ids) == len(metadatas) == len(documents)

    index.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents,
    )

    if provider == 'pinecone': sleep(20)
    assert index.count() == len(ids)
    for group_name in groups:
        # group_ids = index.get(ids=ids, where={"group": group_name}).ids
        # group_ids = index.get(where={"group": group_name}).ids
        get_result = index.get(ids=ids, where={"group": {"$eq":group_name}})
        group_ids = get_result.ids        
        assert group_ids
        assert len(group_ids) == sub_group_size * len(groups[group_name])
    
    query = index.query_many(["dog", "fish"], include=["metadatas", "documents"], top_k=30)
    assert query
    dog_result, fish_result = query
    assert dog_result.ids
    assert dog_result.metadatas
    assert dog_result.documents
    assert "dog" == dog_result.documents[0]
    for doc_species in groups["animals"]["dog"]:
        assert doc_species in dog_result.documents

    assert fish_result.ids
    assert fish_result.metadatas
    assert fish_result.documents
    assert "fish" == fish_result.documents[0]
    for doc_species in groups["animals"]["fish"]:
        assert doc_species in fish_result.documents





    






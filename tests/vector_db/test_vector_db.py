import pytest
from typing import Optional, Literal

from unifai import UnifAI, ProviderName
from unifai.components._base_components._base_vector_db import VectorDB, VectorDBCollection, DocumentDB
from unifai.components import DictDocumentDB

from unifai.exceptions import CollectionNotFoundError, DocumentNotFoundError
from basetest import base_test, base_test_vector_dbs_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS
from chromadb.errors import InvalidCollectionException

from time import sleep
@base_test_vector_dbs_all
def test_init_vector_db_init_dbs(provider, client_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])

    db = ai.get_vector_db(provider)

    assert db
    assert ai._components["vector_db"][provider]["default"] is db
    assert ai.get_vector_db() is db 
    assert ai.get_vector_db(provider) is db 



def parameterize_name_and_metadata(func):
    return pytest.mark.parametrize(
        "name, metadata",
        [
            ("test-collection", {"test": "metadata"}),
            # ("test-collection", {"test": "metadata", "another": "metadata"}),
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
            # ("google", None),
            # ("nvidia", None),
            # ("google", "models/text-embedding-004"),
            # ("google", "embedding-gecko-001"),
            # ("google", "embedding-001"),
            # ("ollama", None),
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
            768,
            # 1536, 
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
            "dotproduct"
        ]
    )(func)









# @base_test(*VECTOR_DB_PROVIDERS, exclude=["pinecone"])
# @base_test_vector_dbs_all
@base_test(*VECTOR_DB_PROVIDERS, exclude=[])
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_create_collection(provider: ProviderName, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[ProviderName],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]],                                                                                               
                                tmp_path,
                                serial
                                ):
    # if provider == "chroma":
    #     client_kwargs["persist_directory"] = str(tmp_path)
    # name = f"{name}_{provider}_{embedding_provider}_{embedding_model}_{dimensions}_{distance_metric}"

    ai = UnifAI(provider_configs=[
        {"provider": provider, "client_init_kwargs": client_kwargs},
        {"provider": "openai", "client_init_kwargs": PROVIDER_DEFAULTS["openai"][1]},
        {"provider": "google", "client_init_kwargs": PROVIDER_DEFAULTS["google"][1]},
        {"provider": "ollama", "client_init_kwargs": PROVIDER_DEFAULTS["ollama"][1]},
    ])


    db = ai.get_vector_db(provider)
    assert db
    assert isinstance(db, VectorDB)
    db.delete_all_collections() # Reset for each test

    collection = db.create_collection(
        name=name,
        # metadata=metadata,
        embedder=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        **func_kwargs
    )
    assert collection
    assert isinstance(collection, VectorDBCollection)
    assert collection.name == name
    

    assert collection.embedding_provider == embedding_provider if embedding_provider is not None else collection.embedder.provider
    assert collection.embedding_model == embedding_model if embedding_model is not None else collection.embedder.default_model
    assert collection.dimensions == dimensions if dimensions is not None else collection.embedder.default_dimensions
    assert collection.distance_metric == db._validate_distance_metric(distance_metric)

    assert db.get_collection(name) is collection

    # assert db.get_collections() == [collection]
    assert db.list_collections() == [name]
    assert db.count_collections() == 1

    collection2_name = "collection-2"
    # TODO both should raise InvalidIndexException
    with pytest.raises(CollectionNotFoundError):
        collection2 = db.get_collection(collection2_name)

    collection2 = db.get_or_create_collection(
        name=collection2_name,
        # metadata=metadata,
        embedder=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        **func_kwargs
    )

    assert collection2
    assert isinstance(collection2, VectorDBCollection)
    assert collection2.name == collection2_name
    # if provider == "chroma": assert collection2.metadata == updated_metadata
    assert collection2.embedding_provider == embedding_provider if embedding_provider is not None else collection2.embedder.provider
    assert collection2.embedding_model == embedding_model if embedding_model is not None else collection2.embedder.default_model
    assert collection2.dimensions == dimensions if dimensions is not None else collection2.embedder.default_dimensions
    assert collection2.distance_metric == db._validate_distance_metric(distance_metric)

    assert db.get_collection(collection2_name) is collection2
    # assert db.list_collections() == [collection2_name, name]
    assert sorted(db.list_collections()) == sorted([name, collection2_name])
    assert db.count_collections() == 2
    
    # test deleting collection
    db.delete_collection(collection2_name)
    assert db.list_collections() == [name]
    assert db.count_collections() == 1
    db.delete_collection(name)
    assert db.list_collections() == []
    assert db.count_collections() == 0
    
    del ai
    

def approx_embeddings(embeddings, expected_embeddings):
    assert len(embeddings) == len(expected_embeddings)
    for i, embedding in enumerate(embeddings):
        for j, value in enumerate(embedding):
            assert pytest.approx(value) == pytest.approx(expected_embeddings[i][j])

# @base_test(*VECTOR_DB_PROVIDERS, exclude=["pinecone"])
# @base_test_vector_dbs_all
@base_test(*VECTOR_DB_PROVIDERS, exclude=[])
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_add(provider: ProviderName, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[ProviderName],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]],                                                                
                                # serial
                                ):

    ai = UnifAI(provider_configs=[
        {"provider": provider, "client_init_kwargs": client_kwargs},
        {"provider": "openai", "client_init_kwargs": PROVIDER_DEFAULTS["openai"][1]},
        {"provider": "google", "client_init_kwargs": PROVIDER_DEFAULTS["google"][1]},
        {"provider": "ollama", "client_init_kwargs": PROVIDER_DEFAULTS["ollama"][1]},
    ])

    db = ai.get_vector_db(provider)
    assert db
    assert isinstance(db, VectorDB)
    db.delete_all_collections() # Reset for each test

    collection = db.create_collection(
        name=name,
        # metadata=metadata,
        embedder=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        document_db_collection="dict",
        **func_kwargs
    )
    assert collection
    assert isinstance(collection, VectorDBCollection)

    collection.add(
        ids=["test_id"],
        metadatas=[{"test": "metadata"}],
        texts=["test document"],
        # embeddings=[Embedding(vector = ([1.0] * dimensions), collection=0)]
    )

    # test including embeddings
    if provider == 'pinecone': sleep(15)
    assert collection.count() == 1
    get_result = collection.get(["test_id"])
    assert get_result
    assert get_result.ids == ["test_id"]
    assert get_result.metadatas == [{"test": "metadata"}]
    assert get_result.texts == ["test document"]
    assert get_result.embeddings == None

    get_result = collection.get(["test_id"], include=["embeddings"])
    assert get_result
    assert get_result.ids == ["test_id"]
    assert get_result.metadatas == None
    assert get_result.texts == None
    assert get_result.embeddings
    assert len(get_result.embeddings) == 1

    computed_embedding = get_result.embeddings[0]

    if dimensions is None:
        dimensions = len(computed_embedding)

    manual_embeddings = [[.1] * dimensions]
    manual_embeddings2 = [[.2] * dimensions]
    

    collection.add(
        ids=["test_id_2"],
        metadatas=[{"test": "metadata2"}],
        texts=["test document2"],
        embeddings=manual_embeddings
    )

    if provider == 'pinecone': sleep(15)
    assert collection.count() == 2
    get_result = collection.get(["test_id_2"], where={"test": "metadata2"}, include=["metadatas", "texts", "embeddings"])
    # get_result = collection.get(where={"test": "metadata2"}, include=["metadatas", "texts", "embeddings"])

    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2"}]
    assert get_result.texts == ["test document2"]
    approx_embeddings(get_result.embeddings, manual_embeddings)

    get_result = collection.get(["test_id", "test_id_2"], include=["metadatas", "texts", "embeddings"])
    assert get_result
    assert get_result.sort().ids == ["test_id", "test_id_2"]
    assert get_result.ids == ["test_id", "test_id_2"]
    assert get_result.metadatas == [{"test": "metadata"}, {"test": "metadata2"}]
    assert get_result.texts == ["test document", "test document2"]
    approx_embeddings(get_result.embeddings, [computed_embedding] + manual_embeddings)

    # test updating
    collection.update(
        ids=["test_id_2"],
        metadatas=[{"test": "metadata2-UPDATED"}],
        texts=["test document2-UPDATED"],
        embeddings=manual_embeddings2
    )

    if provider == 'pinecone': sleep(15)
    assert collection.count() == 2
    get_result = collection.get(["test_id_2"], include=["metadatas", "texts", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2-UPDATED"}]
    assert get_result.texts == ["test document2-UPDATED"]
    approx_embeddings(get_result.embeddings, manual_embeddings2)

    # test deleting
    collection.delete(["test_id_2"])
    if provider == 'pinecone': sleep(15)
    assert collection.count() == 1
    get_result = collection.get(["test_id_2"], include=["metadatas", "texts", "embeddings"])
    assert not get_result.metadatas
    assert not get_result.texts
    assert not get_result.embeddings        

    # test upsert
    collection.upsert(
        ids=["test_id_2"],
        metadatas=[{"test": "metadata2-UPDATED"}],
        texts=["test document2-UPDATED"],
        embeddings=manual_embeddings2
    )

    if provider == 'pinecone': sleep(15)
    assert collection.count() == 2
    get_result = collection.get(["test_id_2"], include=["metadatas", "texts", "embeddings"])
    assert get_result
    assert get_result.ids == ["test_id_2"]
    assert get_result.metadatas == [{"test": "metadata2-UPDATED"}]
    assert get_result.texts == ["test document2-UPDATED"]
    approx_embeddings(get_result.embeddings, manual_embeddings2)

    # Test get/delete all ids
    all_ids = collection.list_ids()
    assert all_ids == ["test_id", "test_id_2"]
    collection.delete_all()
    if provider == 'pinecone': sleep(15)
    assert collection.count() == 0

    # test upsert with multiple
    num_docs = 69
    many_ids, many_metadatas, many_texts, many_embeddings = [], [], [], []
    ids = [(i, f"test_id_{i}") for i in range(num_docs)]
    for i, id in sorted(ids, key=lambda x: x[1]):
        many_ids.append(id)
        many_metadatas.append({"test": f"metadata_{i}"})
        many_texts.append(f"test document_{i}")
        many_embeddings.append([.1] * dimensions)    
        
    collection.add(
        ids=many_ids,
        metadatas=many_metadatas,
        texts=many_texts,
        embeddings=many_embeddings
    )
    if provider == 'pinecone': sleep(60)
    assert collection.count() == num_docs
    get_result = collection.get(many_ids, include=["metadatas", "texts", "embeddings"])
    assert get_result
    print(get_result.ids)
    assert get_result.sort().ids == many_ids
    assert get_result.ids == many_ids
    assert get_result.metadatas == many_metadatas
    assert get_result.texts == many_texts
    approx_embeddings(get_result.embeddings, many_embeddings)

    # test deleting all
    collection.delete_all()
    if provider == 'pinecone': sleep(15)
    assert collection.count() == 0

    # test upsert with multiple after deleting all
    collection.upsert(
        ids=many_ids,
        metadatas=many_metadatas,
        texts=many_texts,
        embeddings=many_embeddings
    )
    if provider == 'pinecone': sleep(60)
    assert collection.count() == num_docs
    get_result = collection.get(many_ids, include=["metadatas", "texts", "embeddings"])
    assert get_result
    print(get_result.ids)
    assert get_result.sort().ids == many_ids
    assert get_result.ids == many_ids
    assert get_result.metadatas == many_metadatas
    assert get_result.texts == many_texts
    approx_embeddings(get_result.embeddings, many_embeddings) 

    del ai

# @base_test_vector_dbs_all
# @base_test(*VECTOR_DB_PROVIDERS, exclude=["pinecone"])])
@base_test(*VECTOR_DB_PROVIDERS, exclude=[])
@parameterize_name_and_metadata
@parameterize_embedding_provider_embedding_model
@parameterize_dimensions
@parameterize_distance_metric
def test_vector_db_query_simple(provider: ProviderName, 
                                client_kwargs: dict, 
                                func_kwargs: dict,
                                name: str, 
                                metadata: dict,
                                embedding_provider: Optional[ProviderName],
                                embedding_model: Optional[str],
                                dimensions: Optional[int],
                                distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]],                                                                
                                # serial
                                ):

    ai = UnifAI(provider_configs=[
        {"provider": provider, "client_init_kwargs": client_kwargs},
        {"provider": "openai", "client_init_kwargs": PROVIDER_DEFAULTS["openai"][1]},
        {"provider": "google", "client_init_kwargs": PROVIDER_DEFAULTS["google"][1]},
        {"provider": "ollama", "client_init_kwargs": PROVIDER_DEFAULTS["ollama"][1]},
    ])

    db = ai.get_vector_db(provider)
    assert db
    assert isinstance(db, VectorDB)
    db.delete_all_collections() # Reset for each test

    collection = db.create_collection(
        name=name,
        # metadata=metadata,
        embedder=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
        distance_metric=distance_metric,
        document_db_collection="dict",
        **func_kwargs
    )
    assert collection
    assert isinstance(collection, VectorDBCollection)

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

    ids, metadatas, texts = [], [], []
    for group_name, group in groups.items():
        for sub_group_name, sub_group in group.items():
            assert len(sub_group) == sub_group_size
            for doc in sub_group:
                ids.append(f"{group_name}_{sub_group_name}_{doc}")
                metadatas.append({"group": group_name, "sub_group": sub_group_name})
                texts.append(doc)

    assert len(ids) == len(metadatas) == len(texts)

    collection.add(
        ids=ids,
        metadatas=metadatas,
        texts=texts,
    )

    if provider == 'pinecone': sleep(20)
    assert collection.count() == len(ids)
    for group_name in groups:
        # group_ids = collection.get(ids=ids, where={"group": group_name}).ids
        # group_ids = collection.get(where={"group": group_name}).ids
        get_result = collection.get(ids=ids, where={"group": {"$eq":group_name}})
        group_ids = get_result.ids        
        assert group_ids
        assert len(group_ids) == sub_group_size * len(groups[group_name])
    
    query = collection.query_many(["dog", "fish"], include=["metadatas", "texts"], top_k=30)
    assert query
    dog_result, fish_result = query
    assert dog_result.ids
    assert dog_result.metadatas
    assert dog_result.texts
    assert "dog" == dog_result.texts[0]
    for doc_species in groups["animals"]["dog"]:
        assert doc_species in dog_result.texts

    assert fish_result.ids
    assert fish_result.metadatas
    assert fish_result.texts
    assert "fish" == fish_result.texts[0]
    for doc_species in groups["animals"]["fish"]:
        assert doc_species in fish_result.texts





    






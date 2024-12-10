import pytest
from typing import Optional, Literal

from unifai import UnifAI, ProviderName
from unifai.components._base_components._base_vector_db import VectorDB, VectorDBCollection
from unifai.components._base_components._base_reranker import Reranker

from unifai.types import ProviderName, GetResult, QueryResult, Embedding, Embeddings, ResponseInfo
from unifai.exceptions import BadRequestError
from basetest import base_test_rerankers, API_KEYS


@base_test_rerankers
def test_init_rerankers(
        provider: ProviderName,
        init_kwargs: dict,
    ):
    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    reranker = ai.reranker(provider)
    assert isinstance(reranker, Reranker)
    assert reranker.provider == provider


@base_test_rerankers
def test_rerank_simple(
        provider: ProviderName,
        init_kwargs: dict,
    ):

    ai = UnifAI(
        api_keys=API_KEYS,
        default_providers={
            "embedder": "openai",
            "vector_db": "chroma",
            "reranker": provider,
        }
    )

    reranker = ai.reranker(provider)
    assert isinstance(reranker, Reranker)
    assert reranker.provider == provider

    texts = [
        'This is a list which containing sample texts.',
        'Keywords are important for keyword-based search.',
        'Document analysis involves extracting keywords.',
        'Keyword-based search relies on sparse embeddings.',
        'Understanding document structure aids in keyword extraction.',
        'Efficient keyword extraction enhances search accuracy.',
        'Semantic similarity improves document retrieval performance.',
        'Machine learning algorithms can optimize keyword extraction methods.'
    ]
    query = "Natural language processing techniques enhance keyword extraction efficiency."

    # collection = ai.get_or_create_collection(
    #     name="reranker_test",
    #     vector_db_provider="chroma",
    #     embedding_provider="openai",
    #     embedding_model="text-embedding-3-large",
    # )
    vector_db = ai.vector_db("chroma")
    vector_db.delete_all_collections() # Clear any existing collections before testing in case previous tests failed to clean up

    collection = vector_db.get_or_create_collection(
        name="reranker_test",
        embedder="openai",
        embedding_model="text-embedding-3-large",
    )

    assert isinstance(collection, VectorDBCollection)
    assert collection.name == "reranker_test"
    assert collection.provider == "chroma"
    assert collection.embedding_provider == "openai"
    assert collection.embedding_model == "text-embedding-3-large"

    assert collection.count() == 0
    doc_ids = [f"doc_{i}" for i in range(len(texts))]
    collection.upsert(
        ids=doc_ids,
        metadatas=[{"doc_collection": i} for i in range(len(texts))],
        texts=texts,
    )
    assert collection.count() == len(texts)


    query_result = collection.query(query_input=query, top_k=6)
    assert isinstance(query_result, QueryResult)
    assert len(query_result) == 6
    assert query_result.ids and query_result.metadatas and query_result.texts

    # Intentionally unorder the query result to force reranker to reorder
    query_result.ids = query_result.ids[::-1]
    query_result.metadatas = query_result.metadatas[::-1]
    query_result.texts = query_result.texts[::-1]

    # Save the original query result (before rerank) for comparison
    old_ids = query_result.ids.copy()
    old_metadatas = query_result.metadatas.copy()
    old_texts = query_result.texts.copy()

    
    # for top_n in (6, 3, 1):
    top_n = 6
    reranked_result = reranker.rerank(
        query=query, 
        query_result=query_result,
        top_n=top_n,
        )
    
    assert isinstance(reranked_result, QueryResult)
    assert len(reranked_result) == top_n
    assert old_ids != reranked_result.ids
    assert old_metadatas != reranked_result.metadatas
    assert old_texts != reranked_result.texts

    assert reranked_result.texts
    for i in range(top_n):
        print(f"Rank {i}:\nOLD {old_ids[i]}: {old_texts[i]}\nNEW {reranked_result.ids[i]}: {reranked_result.texts[i]}\n\n")

    # Reset for next test
    vector_db.delete_all_collections()

    
        
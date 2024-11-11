import pytest
from typing import Optional, Literal

from unifai import UnifAIClient, LLMProvider, VectorDBProvider, Provider, RerankProvider
from unifai.components.retrievers._base_vector_db_client import VectorDBClient, VectorDBIndex
from unifai.components.rerankers._base_reranker import Reranker

from unifai.types import VectorDBProvider, VectorDBGetResult, VectorDBQueryResult, Embedding, Embeddings, ResponseInfo
from unifai.exceptions import BadRequestError
from basetest import base_test_rerankers_all, PROVIDER_DEFAULTS


@base_test_rerankers_all
def test_init_rerankers(
        provider: RerankProvider,
        client_kwargs: dict,
        func_kwargs: dict
    ):
    ai = UnifAIClient({provider: client_kwargs})
    assert ai.provider_client_kwargs == {provider: client_kwargs}
    reranker = ai.get_component(provider, "reranker", **client_kwargs)
    assert isinstance(reranker, Reranker)
    assert reranker.provider == provider
    assert reranker.client_kwargs == client_kwargs
    # assert reranker.client


@base_test_rerankers_all
def test_rerank_simple(
        provider: RerankProvider,
        client_kwargs: dict,
        func_kwargs: dict
    ):

    ai = UnifAIClient({
        provider: client_kwargs,
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "chroma": PROVIDER_DEFAULTS["chroma"][1],
    })

    reranker = ai.get_component(provider, "reranker", **client_kwargs)
    assert isinstance(reranker, Reranker)
    assert reranker.provider == provider
    assert reranker.client_kwargs == client_kwargs    


    documents = [
        'This is a list which containing sample documents.',
        'Keywords are important for keyword-based search.',
        'Document analysis involves extracting keywords.',
        'Keyword-based search relies on sparse embeddings.',
        'Understanding document structure aids in keyword extraction.',
        'Efficient keyword extraction enhances search accuracy.',
        'Semantic similarity improves document retrieval performance.',
        'Machine learning algorithms can optimize keyword extraction methods.'
    ]
    query = "Natural language processing techniques enhance keyword extraction efficiency."

    # index = ai.get_or_create_index(
    #     name="reranker_test",
    #     vector_db_provider="chroma",
    #     embedding_provider="openai",
    #     embedding_model="text-embedding-3-large",
    # )
    vector_db = ai.get_vector_db("chroma")
    vector_db.delete_all_indexes() # Clear any existing indexes before testing in case previous tests failed to clean up

    index = vector_db.get_or_create_index(
        name="reranker_test",
        embedding_provider="openai",
        embedding_model="text-embedding-3-large",
    )

    assert isinstance(index, VectorDBIndex)
    assert index.name == "reranker_test"
    assert index.provider == "chroma"
    assert index.embedding_provider == "openai"
    assert index.embedding_model == "text-embedding-3-large"

    assert index.count() == 0
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    index.upsert(
        ids=doc_ids,
        metadatas=[{"doc_index": i} for i in range(len(documents))],
        documents=documents,
    )
    assert index.count() == len(documents)


    query_result = index.query(query_text=query, top_k=6)
    assert isinstance(query_result, VectorDBQueryResult)
    assert len(query_result) == 6
    assert query_result.ids and query_result.metadatas and query_result.documents

    # Intentionally unorder the query result to force reranker to reorder
    query_result.ids = query_result.ids[::-1]
    query_result.metadatas = query_result.metadatas[::-1]
    query_result.documents = query_result.documents[::-1]

    # Save the original query result (before rerank) for comparison
    old_ids = query_result.ids.copy()
    old_metadatas = query_result.metadatas.copy()
    old_documents = query_result.documents.copy()

    
    # for top_n in (6, 3, 1):
    top_n = 6
    reranked_result = reranker.rerank(
        query=query, 
        query_result=query_result,
        top_n=top_n,
        )
    
    assert isinstance(reranked_result, VectorDBQueryResult)
    assert len(reranked_result) == top_n
    assert old_ids != reranked_result.ids
    assert old_metadatas != reranked_result.metadatas
    assert old_documents != reranked_result.documents

    assert reranked_result.documents
    for i in range(top_n):
        print(f"Rank {i}:\nOLD {old_ids[i]}: {old_documents[i]}\nNEW {reranked_result.ids[i]}: {reranked_result.documents[i]}\n\n")

    # Reset for next test
    vector_db.delete_all_indexes()

    
        
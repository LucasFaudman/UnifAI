import pytest
from typing import Optional, Literal

from unifai import UnifAI, ProviderName
from unifai.components._base_components._base_vector_db import VectorDB, VectorDBCollection
from unifai.components import DictDocumentDB
from unifai.components._base_components._base_reranker import Reranker

from unifai.types import ProviderName, GetResult, QueryResult, Embedding, Embeddings, ResponseInfo
from unifai.exceptions import BadRequestError
from basetest import base_test_rerankers, EMBEDDING_PROVIDERS
from unifai.client.rag_prompter import RAGPromptConfig, RAGPrompter

from time import sleep


@pytest.mark.parametrize("vector_db_provider", [
    "chroma",
    "pinecone",
])
@pytest.mark.parametrize("embedding_provider, embedding_model", [
    ("openai", None),
    ("google", None),
    ("cohere", None),
    ("ollama", None),
    ("nvidia", None),
    ("sentence_transformers", None),
])
@pytest.mark.parametrize("rerank_provider, rerank_model", [
    ("cohere", None),
    ("rank_bm25", None),
    ("sentence_transformers", None),
    ("nvidia", None),
])
def test_rag_engine_simple(
        vector_db_provider: ProviderName,                           
        embedding_provider: ProviderName,
        embedding_model: str,
        rerank_provider: ProviderName,
        rerank_model: str
    ):

    ai = UnifAI({
        vector_db_provider: PROVIDER_DEFAULTS[vector_db_provider][1],
        embedding_provider: PROVIDER_DEFAULTS[embedding_provider][1],
        rerank_provider: PROVIDER_DEFAULTS[rerank_provider][1],
    })


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
    query = "Explain how Natural language processing techniques enhance keyword extraction efficiency."

    vector_db = ai.vector_db(vector_db_provider)
    vector_db.delete_all_indexes() # Clear any existing indexes before testing in case previous tests failed to clean up

    if vector_db_provider == 'pinecone':
        document_db = DictDocumentDB({})
    else:
        document_db = None

    index = vector_db.get_or_create_index(
        name="rag-test",
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        document_db=document_db,
    )

    assert isinstance(index, VectorDBCollection)
    assert index.name == "rag-test"
    assert index.provider == vector_db_provider
    assert index.embedding_provider == embedding_provider
    assert index.embedding_model == embedding_model

    if vector_db_provider == 'pinecone': sleep(10)
    assert index.count() == 0
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    index.upsert(
        ids=doc_ids,
        metadatas=[{"doc_index": i} for i in range(len(documents))],
        documents=documents,
    )

    if vector_db_provider == 'pinecone': sleep(20)
    assert index.count() == len(documents)

    rag_spec = RAGPromptConfig(
        name="test_rag_spec",
        vector_db_index_name="rag-test",
        vector_db_provider=vector_db_provider,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        rerank_provider=rerank_provider,
        rerank_model=rerank_model,
        top_k=5,
        top_n=3,
        document_db_class_or_instance=DictDocumentDB if document_db else None,
        document_db_kwargs={"documents": document_db.documents if document_db else None},
    )

    rag_engine = ai.get_rag_prompter(rag_spec)
    assert isinstance(rag_engine, RAGPrompter)

    query_result = rag_engine.retrieve(query, top_k=5)
    assert query_result
    assert isinstance(query_result, QueryResult)
    assert query_result.ids
    assert query_result.texts
    assert query_result.metadatas
    assert len(query_result) == 5

    rerank_result = rag_engine.rerank(query, query_result, top_n=3)
    assert rerank_result
    assert isinstance(rerank_result, QueryResult)
    assert len(rerank_result) == len(query_result) == 3

    ragify_str = rag_engine.construct_rag_prompt(query, query_result, top_n=3)
    assert ragify_str
    assert isinstance(ragify_str, str)
    print(ragify_str)
    assert "PROMPT:" in ragify_str
    assert "CONTEXT:" in ragify_str
    assert "RESPONSE:" in ragify_str
    assert query in ragify_str
    assert all(doc in ragify_str for doc in query_result.texts)
    assert rerank_result.texts
    assert all(doc in ragify_str for doc in rerank_result.texts)
    assert ragify_str.count("DOCUMENT:") == 3

    ragify_resp = rag_engine.ragify(query)

    assert ragify_resp
    assert ragify_str == ragify_resp
   

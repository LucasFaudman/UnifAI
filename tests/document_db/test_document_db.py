import pytest
from typing import Optional, Literal

from unifai import UnifAI, LLMProvider, VectorDBProvider, Provider
from unifai.components.retrievers._base_vector_db_client import VectorDBClient, VectorDBIndex, DocumentDB
from unifai.components.document_dbs import DocumentDB, DictDocumentDB, SQLiteDocumentDB

from unifai.types import VectorDBProvider, VectorDBGetResult, VectorDBQueryResult, Embedding, Embeddings, ResponseInfo
from unifai.exceptions import BadRequestError, NotFoundError, DocumentNotFoundError
from basetest import base_test, base_test_document_dbs_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS
from chromadb.errors import InvalidCollectionException

@base_test_document_dbs_all
def test_init_document_db_clients(provider, client_kwargs, func_kwargs):
    ai = UnifAI(provider_configs={
        provider: client_kwargs
    })

    client = ai.get_document_db(provider)
    assert isinstance(client, DocumentDB)
    

@base_test_document_dbs_all
def test_get_set_documents(provider, client_kwargs, func_kwargs):
    ai = UnifAI(provider_configs={
        provider: client_kwargs
    })

    client = ai.get_document_db(provider)
    client.set_document("test_id", "test_document")
    assert client.get_document("test_id") == "test_document"
    client.delete_document("test_id")

    with pytest.raises(DocumentNotFoundError):
        client.get_document("test_id")    
    with pytest.raises(NotFoundError):
        client.get_document("test_id")


@base_test_document_dbs_all
@pytest.mark.parametrize("num_documents", 
                         [
                             1, 
                             10, 
                             100, 
                             1000, 
                             10000, 
                             100000
                        ])
def test_many_documents(provider, client_kwargs, func_kwargs, num_documents):
    ai = UnifAI(provider_configs={
        provider: client_kwargs
    })

    client = ai.get_document_db(provider)        

    documents = {f"test_id_{i}": f"test_document_{i}" for i in range(num_documents)}
    client.set_documents(documents.keys(), documents.values())
    for id, document in documents.items():
        assert client.get_document(id) == document
    for id in documents.keys():
        client.delete_document(id)
    for id in documents.keys():
        with pytest.raises(DocumentNotFoundError):
            client.get_document(id)
    
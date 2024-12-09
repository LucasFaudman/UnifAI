import pytest
from typing import Optional, Literal

from unifai import UnifAI
from unifai.components.document_dbs import DocumentDB, DictDocumentDB, SQLiteDocumentDB
from unifai.types import Document

from unifai.exceptions import BadRequestError, NotFoundError, DocumentNotFoundError, CollectionNotFoundError
from basetest import base_test, base_test_document_dbs_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS
from chromadb.errors import InvalidCollectionException

@base_test_document_dbs_all
def test_init_document_db_clients(provider, init_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])

    db = ai.get_document_db(provider)
    assert isinstance(db, DocumentDB)
    

@base_test_document_dbs_all
def test_get_set_documents(provider, init_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])

    db = ai.get_document_db()
    db.upsert(
        collection="default_collection", 
        ids=["test_id"], 
        texts=["test_document"], 
        metadatas=[{"test_key": "test_value"}]
    )

    document = db.get_document(collection="default_collection", id="test_id")
    assert isinstance(document, Document)

    db.delete("default_collection", ["test_id"])
    with pytest.raises(NotFoundError) as e:
        db.get("default_collection", ["test_id"])    
    assert isinstance(e.value, DocumentNotFoundError)

    db.delete_collection("default_collection")
    print(db.list_collections())

    with pytest.raises(NotFoundError) as e:
        db.get_collection("default_collection")
    assert isinstance(e.value, CollectionNotFoundError)
  


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
def test_many_documents(provider, init_kwargs, func_kwargs, num_documents):
    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])

    db = ai.get_document_db(provider)        
    assert isinstance(db, DictDocumentDB)
    documents = {f"test_id_{i}": Document(id=f"test_id_{i}", text=f"test_document_{i}") for i in range(num_documents)}
    
    collection = db.get_or_create_collection("default_collection")
    collection.upsert_documents(documents.values())
    for id, document in documents.items():
        assert collection.get_document(id) == document
    for id in documents.keys():
        collection.delete(id)
    for id in documents.keys():
        with pytest.raises(DocumentNotFoundError):
            collection.get(id)
    for id, document in documents.items():
        collection.upsert(document.id, {"oldkey":"oldval"}, document.text)
        assert collection.get_document(id).metadata == {"oldkey":"oldval"}
    collection.update(list(documents.keys()), [{"newkey":"newval"}]*num_documents)
    for document in collection.get_all_documents():
        assert document.metadata == {"newkey":"newval"}
            
    
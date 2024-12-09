import pytest
from typing import Optional, Literal

from unifai import UnifAI
from unifai.components._base_components._base_document_loader import DocumentLoader, Document
from unifai.components.document_loaders.text_file_loader import TextFileDocumentLoader
from unifai.exceptions import BadRequestError, NotFoundError, DocumentNotFoundError
from basetest import base_test, base_test_document_loaders_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS
from unifai.utils import clean_text


from itertools import zip_longest
from pathlib import Path
RESOURCES_PATH = Path(__file__).parent / "resources"

@base_test_document_loaders_all
def test_init_document_loader_clients(provider, client_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])
    loader = ai.get_document_loader(provider)
    assert isinstance(loader, DocumentLoader)
    assert loader.provider == provider
    

@pytest.mark.parametrize("paths, metadatas, kwargs", [
    (
        (RESOURCES_PATH / "manpages").glob("*"),
        None,
        {}
    ),
])
@base_test("text_file_loader")
def test_text_file_loader(provider, client_kwargs, func_kwargs, paths, metadatas, kwargs):

    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])

    loader = ai.get_document_loader(provider)
    assert isinstance(loader, TextFileDocumentLoader)
    paths = list(paths)

    documents = loader.load_documents(sources=paths, metadatas=metadatas, **kwargs)
    for document, path, metadata in zip_longest(documents, paths, metadatas or ()):
        assert isinstance(document, Document)
        assert isinstance(document.text, str)

        path_text = path.read_text()
        cleaned = clean_text(path_text, loader.config.replacements, loader.config.strip_chars)
        assert document.text == cleaned

        path_str = str(path)
        assert document.id == path_str
        
        if metadata and document.metadata:
            for key, value in metadata.items():
                assert document.metadata[key] == value
            
            if loader.config.add_to_metadata and "source" in loader.config.add_to_metadata:
                assert document.metadata["source"] == path_str
            if loader.config.add_to_metadata and "mimetype" in loader.config.add_to_metadata:
                assert document.metadata["mimetype"] == loader.get_mimetype_with_builtin_mimetypes(path_str)
        
        print(f"Loaded document {document.id}")
        print(document.text[:100])                
        
    assert len(documents) == len(paths)
    print(f"Loaded {len(documents)} documents")







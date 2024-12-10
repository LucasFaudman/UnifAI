import pytest
from typing import Optional, Literal

from unifai import UnifAI
from unifai.components._base_components._base_document_loader import DocumentLoader, Document
from unifai.components.document_loaders.text_file_loader import TextFileDocumentLoader
from unifai.exceptions import BadRequestError, NotFoundError, DocumentNotFoundError

from basetest import base_test, base_test_document_loaders, API_KEYS
from unifai.utils import clean_text


from itertools import zip_longest
from pathlib import Path
RESOURCES_PATH = Path(__file__).parent / "resources"

@base_test_document_loaders
def test_init_document_loader_clients(provider, init_kwargs):
    ai = UnifAI(api_keys=API_KEYS)
    loader = ai.document_loader(provider)
    assert isinstance(loader, DocumentLoader)
    assert loader.provider == provider
    

@pytest.mark.parametrize("paths, metadatas, kwargs", [
    (
        (RESOURCES_PATH / "manpages").glob("*"),
        None,
        {}
    ),
])
def test_text_file_loader(paths, metadatas, kwargs):
    ai = UnifAI(api_keys=API_KEYS)
    
    loader = ai.document_loader("text_file_loader")
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







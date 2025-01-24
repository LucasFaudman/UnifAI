import pytest
from typing import Optional, Literal, Iterable

from unifai import UnifAI, RAGPromptModel
from unifai.components._base_components._base_document_loader import BaseDocumentLoader, DocumentLoader, Document
from unifai.components.document_loaders.text_file_loader import TextFileDocumentLoader
from unifai.configs import DocumentLoaderConfig, FileIODocumentLoaderConfig, RAGConfig
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
    assert isinstance(loader, BaseDocumentLoader)
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




@pytest.mark.parametrize("paths, metadatas, kwargs", [
    (
        (RESOURCES_PATH / "manpages").glob("*"),
        None,
        {}
    ),
])
def test_paramspec_loader(paths, metadatas, kwargs):
    ai = UnifAI(api_keys=API_KEYS)
    
    def load_documents(
            paths: Iterable[Path],
            metadatas: Optional[Iterable[dict|None]] = None,
            some_other_param: Optional[str] = None,
            another_param: Optional[bool] = None
    ):
        for path, metadata in zip_longest(paths, metadatas or ()):
            text = path.read_text()
            metadata = metadata or {}
            metadata.update({"some_other_param": some_other_param, "another_param": another_param})
            yield Document(id=str(path), text=text, metadata=metadata)

    loader = ai.document_loader(
        # DocumentLoaderConfig(
        #     load_documents=load_documents,
        # )     
        # 
        # FileIODocumentLoaderConfig(
        #     provider="text_file_loader",
        #     # load_documents=load_documents,
        # )

        #    
        # "default"        
        # "text_file_loader"
        # ("text_file_loader", "default")
    )
    assert isinstance(loader, BaseDocumentLoader)
    paths = list(paths)

    # documents = loader.load_documents(paths=paths, metadatas=metadatas, some_other_param="test", another_param=True)
    documents = loader.load_documents(sources=paths, metadatas=metadatas, some_other_param="test", another_param=True)

    for document, path, metadata in zip_longest(documents, paths, metadatas or ()):
        assert isinstance(document, Document)
        assert isinstance(document.text, str)

        path_text = path.read_text()
        cleaned = clean_text(path_text, loader.config.replacements, loader.config.strip_chars)
        assert document.text == path_text or document.text == cleaned

        path_str = str(path)
        assert document.id == path_str
        
        if metadata and document.metadata:
            for key, value in metadata.items():
                assert document.metadata[key] == value
            assert document.metadata["some_other_param"] == "test"
            assert document.metadata["another_param"] == True
        
        print(f"Loaded document {document.id}")
        print(document.text[:100])                
        
    assert len(documents) == len(paths)
    print(f"Loaded {len(documents)} documents")

    rag_config = RAGConfig(
        document_loader=load_documents,
        vector_db="chroma"
    )
    ragpipe = ai.ragpipe(rag_config)
    ragpipe.ingest_all(paths=paths, some_other_param="test", another_param=True)
    print(ragpipe.prompt(query="test query"))
    load_documents2 = lambda file_PATHS: load_documents(paths, some_other_param="test", another_param=True)


    ragpipe = (ragpipe
        .set_document_loader(load_documents2) \
        .set_query_modifier(lambda question, end="?": question.strip() + end)
        .set_prompt_template(lambda result, question, tone: f"Question: {question}\nRespond In Tone:{tone}\nManpage Excerpts:\n{result}\nAnswer:")
    )
    
    ragpipe.ingest_all(file_PATHS=paths)
    print(ragpipe.prompt(question="test query", tone="stupid"))


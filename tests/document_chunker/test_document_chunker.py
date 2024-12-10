import pytest
from typing import Optional, Literal
from collections import Counter

from unifai import UnifAI
from unifai.components.document_loaders.text_file_loader import TextFileDocumentLoader
from unifai.components._base_components._base_document_chunker import DocumentChunker, Document
from unifai.exceptions import BadRequestError, NotFoundError, DocumentNotFoundError
from basetest import base_test, base_test_document_chunkers_all, PROVIDER_DEFAULTS, VECTOR_DB_PROVIDERS
from unifai.configs import DocumentChunkerConfig
# from unifai.client.rag_ingestor import RAGIngestor, RAGIngestionConfig
from unifai.configs import RAGConfig
from unifai.components.ragpipe import RAGPipe
from unifai.types.annotations import ProviderName


from pathlib import Path
RESOURCES_PATH = Path(__file__).parent.parent / "document_loader" / "resources"

@base_test_document_chunkers_all
def test_init_document_chunker_clients(provider, init_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    chunker = ai.document_chunker(provider)
    
    assert isinstance(chunker, DocumentChunker)
    assert chunker.provider == provider
    

@base_test_document_chunkers_all
def test_chunk_document_simple(provider, init_kwargs, func_kwargs):
    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    chunker = ai.document_chunker(DocumentChunkerConfig(provider=provider, separators=["\n\n"]))
    
    unchunked_text = '\n\n'.join(f"Chunk me up! - {chunk_num}" * 100 for chunk_num in range(1000))
    unchunked_document = Document(id="doc1", text=unchunked_text)
    chunked_documents = list(chunker.chunk_document(unchunked_document))
    assert len(chunked_documents) == 1000
    for i, chunked_document in enumerate(chunked_documents):
        assert isinstance(chunked_document, Document)
        assert chunked_document.text == f"Chunk me up! - {i}" * 100
        assert chunked_document.id == f"doc1_chunk_{i}"
    print(chunked_documents)



unchunked_document = Document(id="doc1", text='\n\n'.join(f"Chunk me up! - {chunk_num}" * 100 for chunk_num in range(1000)))
loader = TextFileDocumentLoader()
manpages = loader.load_documents((RESOURCES_PATH / "manpages").glob("*"))
imanpages = loader.iload_documents((RESOURCES_PATH / "manpages").glob("*"))

@pytest.mark.parametrize("unchunked_documents", [
    [unchunked_document],
    manpages,
])
@pytest.mark.parametrize("chunk_size", [
    # 69,
    420,
    1337,
    # 8008,    
    # 100,
    1000,
    10000,
    # 100000,
])
@pytest.mark.parametrize("chunk_overlap", [
    0,
    10,
    # 100,
    .1,
    .25,
])
@base_test_document_chunkers_all
def test_size_function_chunkers(provider, init_kwargs, func_kwargs, unchunked_documents, chunk_size, chunk_overlap):
    ai = UnifAI(provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    chunker = ai.document_chunker(provider)
    size_function = chunker.size_function
    # for unchunked_document in unchunked_documents:

    chunked_documents = list(chunker.chunk_documents(unchunked_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    
    chunk_sizes = []
    last_chunk_text = None

    min_chunk = None
    max_chunk = None

    # assert len(chunked_documents) == 1000
    for i, chunked_document in enumerate(chunked_documents):
        assert isinstance(chunked_document, Document)
        # assert chunked_document.text == f"Chunk me up! - {i}" * 100
        # assert chunked_document.id == f"doc1_chunk_{i}"

        assert chunked_document.text        
        curr_chunk_text = chunked_document.text
        curr_chunk_size = size_function(curr_chunk_text)
        chunk_sizes.append(curr_chunk_size)

        # assert curr_chunk_size <= chunk_size
        print(curr_chunk_text, end="\r", flush=True)

        # if chunk_overlap > 0 and last_chunk_text:
        #     assert last_chunk_text[-chunk_overlap:] == curr_chunk_text[:chunk_overlap]
        # last_chunk_text = curr_chunk_text

    min_chunk = sorted(chunked_documents, key=lambda x: size_function(x.text))[0]
    max_chunk = sorted(chunked_documents, key=lambda x: size_function(x.text))[-1]
    max_chunk_size = max(chunk_sizes)
    min_chunk_size = min(chunk_sizes)
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)

    counter = Counter(chunk_sizes)
    for _chunk_size, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print(f"Chunk Size: {_chunk_size} Count: {count}")        

    print("Max chunk_size:", max_chunk_size)
    print("Max Chunk:", max_chunk)
    print()
    print("Min chunk_size:", min_chunk_size)
    print("Min Chunk:", min_chunk)

    print("chunk_size arg:", chunk_size)
    print("Max chunk_size:", max_chunk_size)
    print("Min chunk_size:", min_chunk_size)
    print("Average chunk_size", avg_chunk_size)

    assert max_chunk_size <= chunk_size




def test_ragpipe():
    # ai = UnifAI(
    #     provider_configs={
    #         "openai": PROVIDER_DEFAULTS["openai"][1],
    #         "google": PROVIDER_DEFAULTS["google"][1],
    #         "text_file_loader": PROVIDER_DEFAULTS["text_file_loader"][1],
    #         "count_tokens_chunker": PROVIDER_DEFAULTS["count_tokens_chunker"][1],
    #         "chroma": PROVIDER_DEFAULTS["chroma"][1],
    #         "dict": PROVIDER_DEFAULTS["dict"][1],
    #         "cohere": PROVIDER_DEFAULTS["cohere"][1],
    #         "tiktoken": PROVIDER_DEFAULTS["tiktoken"][1],
    #     })
    ai = UnifAI(provider_configs=[
        {"provider": "openai", "init_kwargs": PROVIDER_DEFAULTS["openai"][1]},
        {"provider": "google", "init_kwargs": PROVIDER_DEFAULTS["google"][1]},
        {"provider": "chroma", "init_kwargs": PROVIDER_DEFAULTS["chroma"][1]},
        {"provider": "cohere", "init_kwargs": PROVIDER_DEFAULTS["cohere"][1]},
    ])

    ragpipe = ai.rag()
    assert isinstance(ragpipe, RAGPipe)


    for i, ingested_doc in enumerate(ragpipe.ingest_documents(manpages)):
        assert isinstance(ingested_doc, Document)
        print("Ingested Document #", i)#, end="\r")

    prompt = 'How can I make a POST request proxied over tor?'
    rag_prompt = ragpipe.prompt(prompt)
    print(rag_prompt)

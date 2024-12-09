from typing import Optional, List, Union, Literal, Iterable, Iterator, Any, Callable, Type, Collection

from .._base_components._base_document_chunker import DocumentChunker

class CountTokensDocumentChunker(DocumentChunker):
    """
    A document chunker that chunks text based on token count.
        chunk_size: Maximum number of tokens in each chunk
        chunk_overlap: Number of tokens to overlap between chunks. Can be an int (number of tokens) or a float (percentage of chunk size)
    Supports both single and multiple separators with recursive splitting.
    """
    provider = "count_tokens_chunker"
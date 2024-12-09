from typing import Optional, List, Union, Literal, Iterable, Iterator, Any, Callable, Type, Collection
from ..tokenizers.str_test_tokenizers import StrLenTokenizer
from .._base_components._base_document_chunker import DocumentChunker

class CharCountDocumentChunker(DocumentChunker):
    provider = "str_len_chunker"

    """
    A document chunker that chunks text based on string length.
    Supports both single and multiple separators with recursive splitting.
    """

    def __init__(
            self,
            chunk_size: int = 420,
            chunk_overlap: int|float = 69,   
            separators: list[str] = ["\n\n", "\n", ""],
            keep_separator: Literal["start", "end", False] = False,            
            regex: bool = False,            
            strip_chars: str|Literal[False] = " \n\t",
            deepcopy_metadata: bool = True,
            add_to_metadata: Optional[Collection[Literal["parent_id", "chunk_size", "start_index", "end_index"]]] = None,
            default_base_id: str = "doc",
            **kwargs
    ) -> None:
        """Initialize the character document chunker.

        Args:
            chunk_size: Maximum size of each chunk in tokens (as measured by the tokenizer)
            chunk_overlap: Number of tokens to overlap between chunks. Can be an int (number of tokens) or a float (percentage of chunk size)
            tokenizer: Tokenizer to use for tokenizing text. Can be an instance of Tokenizer, a subclass of Tokenizer, or a callable that returns the size of a tokenized text
            tokenizer_model: Model to use for the tokenizer
            tokenizer_kwargs: Additional keyword arguments to pass to the tokenizer
            separators: List of character separators to split text on. Default is ["\\n\\n", "\\n", ""]
            keep_separator: Whether and where to keep separators. Options are "start", "end", or False
            regex: Whether to treat separators as regular expressions. Default is False
            strip_chars: Argument to pass to str.strip() to strip characters from the start and end of chunks. Default is " \n\t"
            deepcopy_metadata: Whether to deepcopy metadata for each chunk. Default is True
            add_to_metadata: List of metadata fields to add to each chunk. Options are "parent_id", "chunk_size", "start_index", and "end_index"
            default_base_id: Base ID to use for documents when no ID is provided. Default is "doc"
            **kwargs: Additional keyword arguments for subclasses of DocumentChunker
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=StrLenTokenizer,
            separators=separators,
            keep_separator=keep_separator,
            regex=regex,
            strip_chars=strip_chars,
            deepcopy_metadata=deepcopy_metadata,
            add_to_metadata=add_to_metadata,
            default_base_id=default_base_id,
            **kwargs
        )
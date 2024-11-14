from ._base import UnifAIError

class TokenizerError(UnifAIError):
    """Base class for all tokenizer errors"""

class TokenizerVocabError(TokenizerError):
    """Raised when the tokenizer fails to find a token in the vocabulary"""

class TokenizerDisallowedSpecialTokenError(TokenizerError):
    """
    Raised when a special token is disallowed
    
    Special tokens are artificial tokens used to unlock capabilities from a model,
    such as fill-in-the-middle. So we want to be careful about accidentally encoding special
    tokens, since they can be used to trick a model into doing something we don't want it to do.

    Hence, by default, encode will raise an error if it encounters text that corresponds
    to a special token. This can be controlled on a per-token level using the `allowed_special`
    and `disallowed_special` parameters. In particular:
    - Setting `disallowed_special` to () will prevent this function from raising errors and
        cause all text corresponding to special tokens to be encoded as natural text.
    - Setting `allowed_special` to "all" will cause this function to treat all text
        corresponding to special tokens to be encoded as special tokens.    
    """

# class TokenizerEncodeError(TokenizerError):
#     """Raised when the tokenizer fails to encode the input text"""

# class TokenizerDecodeError(TokenizerError):
#     """Raised when the tokenizer fails to decode the input tokens"""
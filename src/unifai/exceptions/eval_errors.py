from ._base import UnifAIError

class EvalError(UnifAIError):
    """Base class for all Eval errors"""

class OutputParserError(EvalError):
    """Raised when an error occurs while running the output parser"""
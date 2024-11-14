from typing import Optional

class UnifAIError(Exception):
    """Base class for all exceptions in UnifAI"""
    def __init__(self, 
                 message: str, 
                 original_exception: Optional[Exception] = None
                 ):
        self.message = message
        self.original_exception = original_exception
        super().__init__(original_exception)

class UnknownUnifAIError(UnifAIError):
    """Raised when an unknown error occurs in UnifAI"""
    pass
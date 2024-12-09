from typing import Optional, Any

class UnifAITracebackRecord:
    def __init__(self,
                 component: Any,
                 func_name: str,
                 func_args: tuple,
                 func_kwargs: dict,
                 ):
        self.component = component
        self.func_name = func_name
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def __repr__(self):
        return f"UnifAITracebackRecord(component_id={self.component.component_id}, func_name={self.func_name}, func_args={self.func_args}, func_kwargs={self.func_kwargs})"

    def __str__(self):
        return f"Component ID: {self.component.component_id}\nFunction Name: {self.func_name}\nFunction Args: {self.func_args}\nFunction Kwargs: {self.func_kwargs}"

class UnifAIError(Exception):
    """Base class for all exceptions in UnifAI"""
    def __init__(self, 
                 message: str, 
                 original_exception: Optional[Exception] = None
                 ):
        self.message = message
        self.original_exception = original_exception
        self.unifai_traceback: list[UnifAITracebackRecord] = []
        super().__init__(original_exception)

    def add_traceback(
            self, 
            component: Any,
            func_name: str,
            func_args: tuple,
            func_kwargs: dict,
            ):
        self.unifai_traceback.append(UnifAITracebackRecord(component, func_name, func_args, func_kwargs))

    def __str__(self):
        trackback_str = "\n".join(str(tb) for tb in self.unifai_traceback)
        return f"{self.__class__.__name__}: Message: {self.message}\nOriginal Exception: {self.original_exception}\nUnifAI Traceback: {trackback_str}"

class UnknownUnifAIError(UnifAIError):
    """Raised when an unknown error occurs in UnifAI"""
    pass
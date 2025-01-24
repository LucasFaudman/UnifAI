from typing import Optional, Any

TRACEBACK_MAX_LENGTH = 1000

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

    def _func_args_str(self):
        return str(self.func_args)[:TRACEBACK_MAX_LENGTH]
    
    def _func_kwargs_str(self):
        return str(self.func_kwargs)[:TRACEBACK_MAX_LENGTH]

    def __repr__(self):
        return f"UnifAITracebackRecord(component_id={self.component.component_id}, func_name={self.func_name}, func_args={self._func_args_str()}, func_kwargs={self._func_kwargs_str()})"

    def __str__(self):
        return f"Component ID: {self.component.component_id}\nFunction Name: {self.func_name}\nFunction Args: {self._func_args_str()}\nFunction Kwargs: {self._func_kwargs_str()})"

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

    def traceback_str(self):
        return "\n".join([str(record) for record in self.unifai_traceback])

    def __str__(self):
        return f"{self.__class__.__name__}: Message: {self.message}\nOriginal Exception: {repr(self.original_exception)}\nUnifAI Traceback: {self.traceback_str()}"

class UnknownUnifAIError(UnifAIError):
    """Raised when an unknown error occurs in UnifAI"""
    pass
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from .ai_func import AIFunction
from .specs import AgentSpec

from pydantic import BaseModel

class UnifAIAgent:
    def __init__(self, 
                 spec: AgentSpec,
                 *ai_functions: AIFunction, 
                 **kwargs):
        self.functions = {}
        for function in ai_functions:
            self.add_function(function)

    
    def add_function(self, function: AIFunction):
        self.functions[function.name] = function
        setattr(self, function.name, function)

    
    def reset(self):
        for function in self.functions.values():
            function.reset()

    
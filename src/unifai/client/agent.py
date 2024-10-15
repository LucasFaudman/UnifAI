from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

# from .chat import Chat
# from .specs import FuncSpec
# from .rag_engine import RAGEngine
# from ..components.prompt_template import PromptTemplate
# from ..components.output_parser import pydantic_parse
# from ..exceptions import UnifAIError
# from ..types import Message, MessageChunk
# from ..type_conversions import stringify_content

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

    
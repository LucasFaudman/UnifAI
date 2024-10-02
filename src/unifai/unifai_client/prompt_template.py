from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai.types import (
    AIProvider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    ToolCall,
    Usage,
)
from unifai.type_conversions import standardize_tools, standardize_messages, standardize_tool_choice, standardize_response_format


class PromptTemplate:

    def __init__(self, 
                 template: str|Callable[..., str],
                 nested_kwargs: Optional[Mapping[str, Any]] = None,
                 template_kwargs: Optional[Mapping[str, Any]] = None,
                 **kwargs
                 ):
        self.template = template
        self.nested_kwargs = nested_kwargs if nested_kwargs is not None else {}
        self.template_kwargs = template_kwargs if template_kwargs is not None else {}
        self.kwargs = kwargs


    def resolve_kwargs(self, 
                       nested_kwargs: Optional[Mapping[str, Any]] = None,                       
                       **kwargs,
                       ) -> Mapping[str, Any]:  
        nested_kwargs = nested_kwargs if nested_kwargs is not None else {}
        resolved_nested_kwargs = {**self.nested_kwargs, **nested_kwargs}

        resolved_kwargs = {**self.kwargs, **kwargs}
        for key, value in resolved_kwargs.items():            
            if isinstance(value, PromptTemplate):
                resolved_kwargs[key] = value.format(**resolved_nested_kwargs.get(key, {}))
            elif callable(value):
                resolved_kwargs[key] = value(**resolved_nested_kwargs.get(key, {}))

        return resolved_kwargs
    

    def resolve_template(self,
                         template: Callable[..., str],
                         template_kwargs: Optional[Mapping[str, Any]] = None,
                         ) -> str:
        template_kwargs = template_kwargs if template_kwargs is not None else {}
        resolved_template_kwargs = {**self.template_kwargs, **template_kwargs}
        return template(**resolved_template_kwargs)


    def format(self, 
               nested_kwargs: Optional[Mapping[str, Any]] = None,
               template_kwargs: Optional[Mapping[str, Any]] = None,
               **kwargs,               
               ):
        
        resolved_kwargs = self.resolve_kwargs(nested_kwargs, **kwargs)
        if callable(self.template):
            template_str = self.resolve_template(self.template, template_kwargs)
        else:
            template_str = self.template
        return template_str.format(**resolved_kwargs)
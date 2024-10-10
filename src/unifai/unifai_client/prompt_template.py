from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai.types import (
    LLMProvider, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    ToolCall,
    Usage,
)
from unifai.type_conversions import standardize_tools, standardize_messages, standardize_tool_choice, standardize_response_format

from pydantic import BaseModel

class PromptTemplate(BaseModel):
    template: Union[str, Callable[..., str]]
    value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None
    nested_kwargs: Optional[Mapping[str, Any]] = None
    template_getter_kwargs: Optional[Mapping[str, Any]] = None
    kwargs: Optional[Mapping[str, Any]] = None




    def __init__(self, 
                 template: str|Callable[..., str], 
                 value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None,                                
                 nested_kwargs: Optional[Mapping[str, Any]] = None,
                 template_getter_kwargs: Optional[Mapping[str, Any]] = None,                 
                 **kwargs
                 ):
        BaseModel.__init__(self, 
                           template=template, 
                           value_formatters=value_formatters,
                           nested_kwargs=nested_kwargs,
                           template_getter_kwargs=template_getter_kwargs,
                           kwargs=kwargs
                           )

        # self.template = template
        # self.value_formatters = value_formatters if value_formatters is not None else {}
        # self.nested_kwargs = nested_kwargs if nested_kwargs is not None else {}
        # self.template_getter_kwargs = template_getter_kwargs if template_getter_kwargs is not None else {}
        # self.kwargs = kwargs


    def resolve_kwargs(self, 
                       nested_kwargs: Optional[Mapping[str, Any]] = None,                       
                       **kwargs,
                       ) -> dict[str, Any]:  
        # nested_kwargs = nested_kwargs if nested_kwargs is not None else {}
        # resolved_nested_kwargs = {**self.nested_kwargs, **nested_kwargs}
        resolved_nested_kwargs = {**self.nested_kwargs} if self.nested_kwargs else {}
        if nested_kwargs:
            resolved_nested_kwargs.update(nested_kwargs)

        # resolved_kwargs = {**self.kwargs, **kwargs}
        resolved_kwargs = {**self.kwargs} if self.kwargs else {}
        if kwargs:
            resolved_kwargs.update(kwargs)

        for key, value in resolved_kwargs.items():
            if isinstance(value, PromptTemplate):
                resolved_kwargs[key] = value.format(**resolved_nested_kwargs.get(key, {}))
            elif callable(value):
                resolved_kwargs[key] = value(**resolved_nested_kwargs.get(key, {}))

        return resolved_kwargs
    

    def resolve_template(self,
                         template: Callable[..., str],
                         template_getter_kwargs: Optional[Mapping[str, Any]] = None,
                         ) -> str:
        # resolved_template_getter_kwargs = {**self.template_getter_kwargs, **(template_getter_kwargs or {})}
        resolved_template_getter_kwargs = {**self.template_getter_kwargs} if self.template_getter_kwargs else {}
        if template_getter_kwargs:
            resolved_template_getter_kwargs.update(template_getter_kwargs)

        return template(**resolved_template_getter_kwargs)


    def format_values(self, 
                      resolved_kwargs: dict[str, Any],
                      value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None,                                      
                      ):
        # value_formatters = {**self.value_formatters, **(value_formatters or {})}
        resolved_value_formatters = {**self.value_formatters} if self.value_formatters else {}
        if value_formatters:
            resolved_value_formatters.update(value_formatters)
        
        for key, value in resolved_kwargs.items():       
            if formatter := resolved_value_formatters.get(key) or resolved_value_formatters.get(type(value)):
                resolved_kwargs[key] = formatter(value)
        return resolved_kwargs
    

    def format(self, 
               nested_kwargs: Optional[Mapping[str, Any]] = None,
               template_getter_kwargs: Optional[Mapping[str, Any]] = None,
               value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None,               
               **kwargs,               
               ):
        
        if callable(self.template):
            template_str = self.resolve_template(self.template, template_getter_kwargs)
        else:
            template_str = self.template

        resolved_kwargs = self.resolve_kwargs(nested_kwargs, **kwargs)
        resolved_kwargs = self.format_values(resolved_kwargs, value_formatters)
        return template_str.format(**resolved_kwargs)
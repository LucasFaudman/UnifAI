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
    template: str|Callable[..., str] = "{content}"
    value_formatters: Optional[Mapping[str|type, Optional[Callable[..., Any]]]] = None
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
        
        # Note to self reason for this is to allow PromptTemplate("Hello {name}", name="World")
        # to work as PromptTemplate(template="Hello {name}", name="World")
        # which is not possible with pydantic BaseModel which requires PromptTemplate(template="Hello {name}")
        BaseModel.__init__(self, 
                           template=template, 
                           value_formatters=value_formatters,
                           nested_kwargs=nested_kwargs,
                           template_getter_kwargs=template_getter_kwargs,
                           kwargs=kwargs
                           )


    def resolve_kwargs(self, 
                       nested_kwargs: Optional[Mapping[str, Any]] = None,                       
                       **kwargs,
                       ) -> dict[str, Any]:
        
        # Comhine nested_kwargs passed on init and nested_kwargs passed on format
        resolved_nested_kwargs = {**self.nested_kwargs} if self.nested_kwargs else {}
        if nested_kwargs:
            resolved_nested_kwargs.update(nested_kwargs)
        # Same as above but for kwargs
        resolved_kwargs = {**self.kwargs} if self.kwargs else {}
        if kwargs:
            resolved_kwargs.update(kwargs)

        for key, value in resolved_kwargs.items():
            if callable(value):
                resolved_kwargs[key] = value(**resolved_nested_kwargs.get(key, {}))
        
        return resolved_kwargs
    

    def resolve_template(self,
                         template: Callable[..., str],
                         template_getter_kwargs: Optional[Mapping[str, Any]] = None,
                         ) -> str:
        
        # Comhine template_getter_kwargs passed on init and template_getter_kwargs passed on format
        resolved_template_getter_kwargs = {**self.template_getter_kwargs} if self.template_getter_kwargs else {}
        if template_getter_kwargs:
            resolved_template_getter_kwargs.update(template_getter_kwargs)

        return template(**resolved_template_getter_kwargs)


    def format_values(self, 
                      resolved_kwargs: dict[str, Any],
                      value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None,                                      
                      ):
        
        resolved_value_formatters = {**self.value_formatters} if self.value_formatters else {}
        if value_formatters:
            resolved_value_formatters.update(value_formatters)
        
        global_formatter = resolved_value_formatters.get("*")
        for key, value in resolved_kwargs.items():       
            if formatter := (
                resolved_value_formatters.get(key) 
                or resolved_value_formatters.get(type(value))
                or global_formatter
                ):
                resolved_kwargs[key] = formatter(value)
        return resolved_kwargs
    

    def format(self, 
               nested_kwargs: Optional[Mapping[str, Any]] = None,
               template_getter_kwargs: Optional[Mapping[str, Any]] = None,
               value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None,               
               **kwargs,               
               ):
        
        if callable(self.template):
            # Allow template to be a callable that returns a string
            template_str = self.resolve_template(self.template, template_getter_kwargs)
        else:
            template_str = self.template
        resolved_kwargs = self.resolve_kwargs(nested_kwargs, **kwargs)
        resolved_kwargs = self.format_values(resolved_kwargs, value_formatters)
        return template_str.format(**resolved_kwargs)
    

    def __call__(self,
                    nested_kwargs: Optional[Mapping[str, Any]] = None,
                    template_getter_kwargs: Optional[Mapping[str, Any]] = None,
                    value_formatters: Optional[Mapping[str|type, Callable[..., Any]]] = None,                 
                    **kwargs,
                    ):
            return self.format(nested_kwargs, template_getter_kwargs, value_formatters, **kwargs)


    def __str__(self):
        return self.format()
    

if __name__ == "__main__":
    hello = PromptTemplate(template="Hello {name}")
    print(hello(name="World")) # Prints Hello World

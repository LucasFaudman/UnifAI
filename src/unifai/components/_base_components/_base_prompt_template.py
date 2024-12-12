from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator
from pydantic import BaseModel
from ...utils import combine_dicts

class PromptTemplate(BaseModel):
    template: str|Callable[..., str] = "{content}"
    value_formatters: Optional[dict[str|type, Optional[Callable[..., Any]]]] = None
    nested_kwargs: Optional[dict[str, Any]] = None
    template_getter_kwargs: Optional[dict[str, Any]] = None
    kwargs: Optional[dict[str, Any]] = None


    def __init__(self, 
                 template: str|Callable[..., str], 
                 value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,                                
                 nested_kwargs: Optional[dict[str, Any]] = None,
                 template_getter_kwargs: Optional[dict[str, Any]] = None,                 
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
                       nested_kwargs: Optional[dict[str, Any]] = None,
                       **kwargs,
                       ) -> dict[str, Any]:        
        # Combine kwargs/nested_kwargs passed on init with nested_kwargs passed on format
        resolved_nested_kwargs = combine_dicts(self.nested_kwargs, nested_kwargs)
        resolved_kwargs = combine_dicts(self.kwargs, kwargs)
        for key, value in resolved_kwargs.items():
            if callable(value):
                resolved_kwargs[key] = value(**resolved_nested_kwargs.get(key, {}))      
        return resolved_kwargs
    
    def resolve_template(self,
                         template: Callable[..., str],
                         template_getter_kwargs: Optional[dict[str, Any]] = None,
                         ) -> str:
        return template(**combine_dicts(self.template_getter_kwargs, template_getter_kwargs))

    def format_values(self, 
                      resolved_kwargs: dict[str, Any],
                      value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,                                      
                      ):        
        resolved_value_formatters = combine_dicts(self.value_formatters, value_formatters)        
        global_formatter = resolved_value_formatters.get("*")
        for key, value in resolved_kwargs.items():       
            if key_formatter := resolved_value_formatters.get(key):
                resolved_kwargs[key] = key_formatter(value)
                continue
            if type_formatter := resolved_value_formatters.get(type(value)):
                resolved_kwargs[key] = type_formatter(value)
                continue            
            used_parent_type_formatter = False
            for formatter_key, parent_type_formatter in resolved_value_formatters.items():
                if isinstance(formatter_key, type) and isinstance(value, formatter_key):
                    resolved_kwargs[key] = parent_type_formatter(value)
                    used_parent_type_formatter = True
                    break
            if used_parent_type_formatter:
                continue
            if global_formatter:
                resolved_kwargs[key] = global_formatter(value)
                continue
            
        return resolved_kwargs
    
    def format(self, 
               nested_kwargs: Optional[dict[str, Any]] = None,
               template_getter_kwargs: Optional[dict[str, Any]] = None,
               value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,               
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
                    nested_kwargs: Optional[dict[str, Any]] = None,
                    template_getter_kwargs: Optional[dict[str, Any]] = None,
                    value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,                 
                    **kwargs,
                    ):
            return self.format(nested_kwargs, template_getter_kwargs, value_formatters, **kwargs)

    def __str__(self):
        return self.format()

    
if __name__ == "__main__":
    hello = PromptTemplate(template="Hello {name}")
    print(hello(name="World")) # Prints Hello World

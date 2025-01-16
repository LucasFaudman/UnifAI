from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator, ClassVar
from textwrap import dedent

from ...types import BaseModel, Field
from ...utils import combine_dicts

from warnings import filterwarnings
filterwarnings("ignore", message=r"Field name \".*\" in \"\w*Prompt\w*\" shadows an attribute in parent \"\w*Prompt\w*\"")

class PromptModel(BaseModel):
    template: ClassVar[str | Callable[..., str] | Literal["__doc__"]] = "__doc__"
    template_getter_kwargs: ClassVar[Optional[dict[str, Any]]] = None

    default_kwargs: ClassVar[Optional[dict[str, Any]]] = None
    default_nested_kwargs: ClassVar[Optional[dict[str, Any]]] = None

    dump_instance: ClassVar[bool] = True
    exclude: ClassVar[set[str]] = {"template", "value_formatters", "default_kwargs", "default_nested_kwargs", "template_getter_kwargs"}

    value_formatters: ClassVar[Optional[dict[str|type, Optional[Callable[..., Any]]]]] = None

    def resolve_template(self,
                         template_getter_kwargs: Optional[dict[str, Any]] = None,
                         ) -> str:
        if self.template == "__doc__":
            if not self.__doc__ or not (_template := dedent(self.__doc__)):
                raise ValueError("Prompt subclass must have a docstring to format")
            # Set template for instance to docstring if first time called
            self.__class__.template = _template
            return _template
        elif callable(self.template):
            return self.template(**combine_dicts(self.template_getter_kwargs, template_getter_kwargs))            
        return self.template       

    def resolve_kwargs(self,
                       nested_kwargs: Optional[dict[str, Any]] = None,
                       **kwargs,
                       ) -> dict[str, Any]:        
        # Combine kwargs/nested_kwargs passed on init with nested_kwargs passed on format
        dumpped_kwargs = self.model_dump(mode="python", exclude=self.exclude) if self.dump_instance else None
        resolved_kwargs = combine_dicts(self.default_kwargs, dumpped_kwargs, kwargs)
        resolved_nested_kwargs = combine_dicts(self.default_nested_kwargs, nested_kwargs)
        for key, value in resolved_kwargs.items():
            if callable(value):
                resolved_kwargs[key] = value(**resolved_nested_kwargs.get(key, {}))
        return resolved_kwargs
    
    def format_values(self, 
                      value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,  
                      **resolved_kwargs: dict[str, Any],                                    
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
               value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,          
               template_getter_kwargs: Optional[dict[str, Any]] = None,
               **kwargs,               
               ) -> str:
        # Resolve template string first so errors are raised before processing kwargs
        template_str = self.resolve_template(template_getter_kwargs)
        resolved_kwargs = self.resolve_kwargs(nested_kwargs, **kwargs)
        formatted_kwargs = self.format_values(value_formatters, **resolved_kwargs)        
        return template_str.format(**formatted_kwargs)

    __call__ = format

    def __str__(self) -> str:
        return self.format()
    

class PromptTemplate(PromptModel):
    "{input}{content}"
    template: str | Callable[..., str] | Literal["__doc__"]  = Field(default="__doc__")
    template_getter_kwargs: Optional[dict[str, Any]] = None

    default_kwargs: Optional[dict[str, Any]] = None
    default_nested_kwargs: Optional[dict[str, Any]] = None

    value_formatters: Optional[dict[str|type, Optional[Callable[..., Any]]]] = None

    def __init__(self, 
                 template: str|Callable[..., str], 
                 value_formatters: Optional[dict[str|type, Callable[..., Any]]] = None,                                
                 template_getter_kwargs: Optional[dict[str, Any]] = None,
                 default_nested_kwargs: Optional[dict[str, Any]] = None,
                 **default_kwargs,
                 ):
        
        # Note to self reason for this is to allow PromptTemplate("Hello {name}", name="World")
        # to work as PromptTemplate(template="Hello {name}", name="World")
        # which is not possible with pydantic BaseModel which requires PromptTemplate(template="Hello {name}")
        init_kwargs = {}
        if template is not None:
            init_kwargs["template"] = template
        if value_formatters is not None:
            init_kwargs["value_formatters"] = value_formatters
        if template_getter_kwargs is not None:
            init_kwargs["template_getter_kwargs"] = template_getter_kwargs
        if default_kwargs is not None:
            init_kwargs["default_kwargs"] = default_kwargs
        if default_nested_kwargs is not None:
            init_kwargs["default_nested_kwargs"] = default_nested_kwargs
        if default_kwargs:
            init_kwargs["default_kwargs"] = default_kwargs
        BaseModel.__init__(self, **init_kwargs)


if __name__ == "__main__":
    hello = PromptTemplate(template="Hello {name}")
    print(hello(name="World")) # Prints Hello World
    class HelloPrompt(PromptModel):
        "Hello {name}"
        name: str

    hello = HelloPrompt(name="World")
    print(hello) # Prints Hello World

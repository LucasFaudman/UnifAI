import pytest

from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai import UnifAIClient, AIProvider, PromptTemplate
from unifai.types import Message, Tool
from basetest import base_test_all_providers

from datetime import datetime

def some_func(kwarg1=None, kwarg2=None):
    return f"{kwarg1=} {kwarg2=}"

def get_time(format="%Y-%m-%d %H:%M:%S"):
    return datetime.now().strftime(format)

def get_template(template_type: str):
    template_kwarg2 = "{template1_kwarg2}" if template_type == "template_type1" else "{template2_kwarg2}"
    return f"Template type: {template_type} {{template_kwarg1}} {template_kwarg2}"    

class ClassWithDunderFormat:
    def __init__(self, value: float|int):
        self.value = float(value) if isinstance(value, int) else value

    def __str__(self):
        return f"string_value={self.value}"

    def __repr__(self):
        return f"ClassWithDunderFormat[value={self.value}]"
    
    def __format__(self, format_spec):
        if format_spec == "!r":
            return repr(self)
        if format_spec == "!s":
            return str(self)
        return f"{self.value:{format_spec}}"
        
    

def _test_prompt_template(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_kwargs: Optional[dict], 
                         expected: str
                         ):
    prompt = PromptTemplate(template, 
                            nested_kwargs=init_nested_kwargs, 
                            template_kwargs=init_template_kwargs,
                            **init_kwargs
                            )
    assert prompt.template == template
    assert prompt.nested_kwargs == (init_nested_kwargs if init_nested_kwargs is not None else {})
    assert prompt.template_kwargs == (init_template_kwargs if init_template_kwargs is not None else {})
    assert prompt.kwargs == init_kwargs

    formatted = prompt.format(
        nested_kwargs=call_nested_kwargs,
        template_kwargs=call_template_kwargs,
        **call_kwargs
        )     
    print(f"{formatted=}")
    assert formatted == expected


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_kwargs, call_kwargs, call_nested_kwargs, call_template_kwargs, expected", [
    (
        """Test template {str_value}""", # template
        {"str_value": "string"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template string" # expected
    ),
    (
        """Test template {str_value}""", # template
        {"str_value": "string"}, # init_kwargs
        None, # init_nested_kwargs
        None, # init_template_kwargs
        {}, # call_kwargs
        None, # call_nested_kwargs
        None, # call_template_kwargs
        "Test template string" # expected
    ), 
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"str_value": "string"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template string" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        None, # init_nested_kwargs
        None, # init_template_kwargs
        {"str_value": "string"}, # call_kwargs
        None, # call_nested_kwargs
        None, # call_template_kwargs
        "Test template string" # expected
    ),    
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"str_value": "string"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template string" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        None, # init_nested_kwargs
        None, # init_template_kwargs
        {"str_value": "string"}, # call_kwargs
        None, # call_nested_kwargs
        None, # call_template_kwargs
        "Test template string" # expected
    ),  

])
def test_prompt_template_simple(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template, 
                         init_kwargs, 
                         init_nested_kwargs, 
                         init_template_kwargs, 
                         call_kwargs, 
                         call_nested_kwargs, 
                         call_template_kwargs, 
                         expected
                         )
    


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_kwargs, call_kwargs, call_nested_kwargs, call_template_kwargs, expected", [ 
    (
        """Test template {str_value} {float_value} {float_value_fmted:.2f} {cls_w_fmt:!r}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {
            "str_value": "string",
            "float_value": 4.2069,
            "float_value_fmted": 6.9696,
            "cls_w_fmt": ClassWithDunderFormat(420.6969)
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template string 4.2069 6.97 ClassWithDunderFormat[value=420.6969]" # expected
    ),
    (
        """Test template {str_value} {float_value} {float_value_fmted:.2f} {cls_w_fmt:!r}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {
            "str_value": "string",
            "float_value": 4.2069,
            "float_value_fmted": 6.9696,
            "cls_w_fmt": ClassWithDunderFormat(420.6969)
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template string 4.2069 6.97 ClassWithDunderFormat[value=420.6969]" # expected
    ),
    (
        """Test template {cls_r:!r} {cls_s:!s} {cls_2f:.2f} {cls_4f:.4f}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {
            "cls_r": ClassWithDunderFormat(420.6969),
            "cls_s": ClassWithDunderFormat(420.6969),
            "cls_2f": ClassWithDunderFormat(420.6969),
            "cls_4f": ClassWithDunderFormat(420.6969)
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template ClassWithDunderFormat[value=420.6969] string_value=420.6969 420.70 420.6969" # expected
    ),
])
def test_prompt_template_format_specifiers(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template, 
                         init_kwargs, 
                         init_nested_kwargs, 
                         init_template_kwargs, 
                         call_kwargs, 
                         call_nested_kwargs, 
                         call_template_kwargs, 
                         expected
                         )    
    


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_kwargs, call_kwargs, call_nested_kwargs, call_template_kwargs, expected", [
    (
        """Test template {str_value}""", # template
        {"str_value": some_func}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template kwarg1=None kwarg2=None" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"str_value": some_func}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template kwarg1=None kwarg2=None" # expected
    ),
    (
        """Test template {str_value}""", # template
        {"str_value": some_func}, # init_kwargs
        {"str_value": {"kwarg1": "nested1", "kwarg2": "nested2"}}, # init_nested_kwargs
        {}, # init_template_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template kwarg1='nested1' kwarg2='nested2'" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"str_value": some_func}, # call_kwargs
        {"str_value": {"kwarg1": "nested1", "kwarg2": "nested2"}}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Test template kwarg1='nested1' kwarg2='nested2'" # expected
    ),    

])
def test_prompt_template_callables(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template, 
                         init_kwargs, 
                         init_nested_kwargs, 
                         init_template_kwargs, 
                         call_kwargs, 
                         call_nested_kwargs, 
                         call_template_kwargs, 
                         expected
                         )    


nested_template = PromptTemplate("Nested template {template_kwarg1} {template_kwarg2}")
nested_template_with_callable = PromptTemplate("Nested template {template_kwarg1}", template_kwarg1=some_func)

@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_kwargs, call_kwargs, call_nested_kwargs, call_template_kwargs, expected", [
    (
        """Parent template {parent_kwarg} {nested_template}""", # template
        {"parent_kwarg": "parent_value", "nested_template": nested_template}, # init_kwargs
        {"nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}}, # init_nested_kwargs
        {}, # init_template_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Parent template parent_value Nested template template_val1 template_val2" # expected
    ),
    (
        """Parent template {parent_kwarg} {nested_template}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"parent_kwarg": "parent_value", "nested_template": nested_template}, # call_kwargs
        {"nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Parent template parent_value Nested template template_val1 template_val2" # expected
    ),    
    (
        """Parent template {parent_kwarg} {parent_func} {nested_template}""", # template
        {"parent_kwarg": "parent_value", "parent_func": some_func, "nested_template": nested_template}, # init_kwargs
        {
            "parent_func": {"kwarg1": "nested1", "kwarg2": "nested2"},
            "nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}
        }, # init_nested_kwargs
        {}, # init_template_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Parent template parent_value kwarg1='nested1' kwarg2='nested2' Nested template template_val1 template_val2" # expected
    ),
    (
        """Parent template {parent_kwarg} {parent_func} {nested_template}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"parent_kwarg": "parent_value", "parent_func": some_func, "nested_template": nested_template}, # call_kwargs
        {
            "parent_func": {"kwarg1": "nested1", "kwarg2": "nested2"},
            "nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Parent template parent_value kwarg1='nested1' kwarg2='nested2' Nested template template_val1 template_val2" # expected
    ),   

])
def test_prompt_template_nested_templates(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template, 
                         init_kwargs, 
                         init_nested_kwargs, 
                         init_template_kwargs, 
                         call_kwargs, 
                         call_nested_kwargs, 
                         call_template_kwargs, 
                         expected
                         )     
    

@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_kwargs, call_kwargs, call_nested_kwargs, call_template_kwargs, expected", [
    (
        get_template, # template
        {"template_kwarg1": "template1_val1"}, # init_kwargs
        {}, # init_nested_kwargs
        {"template_type": "template_type1"}, # init_template_kwargs
        {"template1_kwarg2": "template1_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Template type: template_type1 template1_val1 template1_val2" # expected
    ),
    (
        get_template, # template
        {"template_kwarg1": "template_val2"}, # init_kwargs
        {}, # init_nested_kwargs
        {"template_type": "template_type2"}, # init_template_kwargs
        {"template2_kwarg2": "template2_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_kwargs
        "Template type: template_type2 template_val2 template2_val2" # expected
    ),  
    (
        get_template, # template
        {"template_kwarg1": "template1_val1"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"template1_kwarg2": "template1_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {"template_type": "template_type1"}, # call_template_kwargs
        "Template type: template_type1 template1_val1 template1_val2" # expected
    ),
    (
        get_template, # template
        {"template_kwarg1": "template_val2"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_kwargs
        {"template2_kwarg2": "template2_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {"template_type": "template_type2"}, # call_template_kwargs
        "Template type: template_type2 template_val2 template2_val2" # expected
    ),      


])
def test_prompt_template_template_callable(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template, 
                         init_kwargs, 
                         init_nested_kwargs, 
                         init_template_kwargs, 
                         call_kwargs, 
                         call_nested_kwargs, 
                         call_template_kwargs, 
                         expected
                         )



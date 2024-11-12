import pytest

from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Self, Iterable, Mapping, Generator

from unifai import UnifAI, LLMProvider, PromptTemplate
from unifai.types import Message, Tool
from basetest import base_test_llms_all

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
                         init_kwargs: dict = {}, 
                         init_nested_kwargs: Optional[dict] = None, 
                         init_template_getter_kwargs: Optional[dict] = None, 
                         init_value_formatters: Optional[dict] = None,
                         call_kwargs: dict = {}, 
                         call_nested_kwargs: Optional[dict] = None, 
                         call_template_getter_kwargs: Optional[dict] = None, 
                         call_value_formatters: Optional[dict] = None,
                         expected: str = ""
                         ):
    prompt = PromptTemplate(template, 
                            nested_kwargs=init_nested_kwargs, 
                            template_getter_kwargs=init_template_getter_kwargs,
                            value_formatters=init_value_formatters,
                            **init_kwargs
                            )
    assert prompt.template == template
    assert prompt.nested_kwargs == init_nested_kwargs
    assert prompt.template_getter_kwargs == init_template_getter_kwargs
    assert prompt.kwargs == init_kwargs

    formatted = prompt.format(
        nested_kwargs=call_nested_kwargs,
        template_getter_kwargs=call_template_getter_kwargs,
        value_formatters=call_value_formatters,
        **call_kwargs
        )     
    print(f"{formatted=}")
    assert formatted == expected


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, expected", [
    (
        """Test template {str_value}""", # template
        {"str_value": "string"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template string" # expected
    ),
    (
        """Test template {str_value}""", # template
        {"str_value": "string"}, # init_kwargs
        None, # init_nested_kwargs
        None, # init_template_getter_kwargs
        {}, # call_kwargs
        None, # call_nested_kwargs
        None, # call_template_getter_kwargs
        "Test template string" # expected
    ), 
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_value": "string"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template string" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        None, # init_nested_kwargs
        None, # init_template_getter_kwargs
        {"str_value": "string"}, # call_kwargs
        None, # call_nested_kwargs
        None, # call_template_getter_kwargs
        "Test template string" # expected
    ),    
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_value": "string"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template string" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        None, # init_nested_kwargs
        None, # init_template_getter_kwargs
        {"str_value": "string"}, # call_kwargs
        None, # call_nested_kwargs
        None, # call_template_getter_kwargs
        "Test template string" # expected
    ),  

])
def test_prompt_template_simple(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template=template, 
                         init_kwargs=init_kwargs,
                         init_nested_kwargs=init_nested_kwargs, 
                         init_template_getter_kwargs=init_template_getter_kwargs, 
                         init_value_formatters=None,
                         call_kwargs=call_kwargs, 
                         call_nested_kwargs=call_nested_kwargs, 
                         call_template_getter_kwargs=call_template_getter_kwargs, 
                         call_value_formatters=None,
                         expected=expected
                         )
    


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, expected", [ 
    (
        """Test template {str_value} {float_value} {float_value_fmted:.2f} {cls_w_fmt:!r}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {
            "str_value": "string",
            "float_value": 4.2069,
            "float_value_fmted": 6.9696,
            "cls_w_fmt": ClassWithDunderFormat(420.6969)
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template string 4.2069 6.97 ClassWithDunderFormat[value=420.6969]" # expected
    ),
    (
        """Test template {str_value} {float_value} {float_value_fmted:.2f} {cls_w_fmt:!r}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {
            "str_value": "string",
            "float_value": 4.2069,
            "float_value_fmted": 6.9696,
            "cls_w_fmt": ClassWithDunderFormat(420.6969)
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template string 4.2069 6.97 ClassWithDunderFormat[value=420.6969]" # expected
    ),
    (
        """Test template {cls_r:!r} {cls_s:!s} {cls_2f:.2f} {cls_4f:.4f}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {
            "cls_r": ClassWithDunderFormat(420.6969),
            "cls_s": ClassWithDunderFormat(420.6969),
            "cls_2f": ClassWithDunderFormat(420.6969),
            "cls_4f": ClassWithDunderFormat(420.6969)
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template ClassWithDunderFormat[value=420.6969] string_value=420.6969 420.70 420.6969" # expected
    ),
])
def test_prompt_template_format_specifiers(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template=template, 
                         init_kwargs=init_kwargs,
                         init_nested_kwargs=init_nested_kwargs, 
                         init_template_getter_kwargs=init_template_getter_kwargs, 
                         init_value_formatters=None,
                         call_kwargs=call_kwargs, 
                         call_nested_kwargs=call_nested_kwargs, 
                         call_template_getter_kwargs=call_template_getter_kwargs, 
                         call_value_formatters=None,
                         expected=expected
                         ) 
    


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, expected", [
    (
        """Test template {str_value}""", # template
        {"str_value": some_func}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template kwarg1=None kwarg2=None" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_value": some_func}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template kwarg1=None kwarg2=None" # expected
    ),
    (
        """Test template {str_value}""", # template
        {"str_value": some_func}, # init_kwargs
        {"str_value": {"kwarg1": "nested1", "kwarg2": "nested2"}}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template kwarg1='nested1' kwarg2='nested2'" # expected
    ),
    (
        """Test template {str_value}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_value": some_func}, # call_kwargs
        {"str_value": {"kwarg1": "nested1", "kwarg2": "nested2"}}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Test template kwarg1='nested1' kwarg2='nested2'" # expected
    ),    

])
def test_prompt_template_callables(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template=template, 
                         init_kwargs=init_kwargs,
                         init_nested_kwargs=init_nested_kwargs, 
                         init_template_getter_kwargs=init_template_getter_kwargs, 
                         init_value_formatters=None,
                         call_kwargs=call_kwargs, 
                         call_nested_kwargs=call_nested_kwargs, 
                         call_template_getter_kwargs=call_template_getter_kwargs, 
                         call_value_formatters=None,
                         expected=expected
                         ) 


nested_template = PromptTemplate("Nested template {template_kwarg1} {template_kwarg2}")
nested_template_with_callable = PromptTemplate("Nested template {template_kwarg1}", template_kwarg1=some_func)

@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, expected", [
    (
        """Parent template {parent_kwarg} {nested_template}""", # template
        {"parent_kwarg": "parent_value", "nested_template": nested_template}, # init_kwargs
        {"nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Parent template parent_value Nested template template_val1 template_val2" # expected
    ),
    (
        """Parent template {parent_kwarg} {nested_template}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"parent_kwarg": "parent_value", "nested_template": nested_template}, # call_kwargs
        {"nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Parent template parent_value Nested template template_val1 template_val2" # expected
    ),    
    (
        """Parent template {parent_kwarg} {parent_func} {nested_template}""", # template
        {"parent_kwarg": "parent_value", "parent_func": some_func, "nested_template": nested_template}, # init_kwargs
        {
            "parent_func": {"kwarg1": "nested1", "kwarg2": "nested2"},
            "nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}
        }, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Parent template parent_value kwarg1='nested1' kwarg2='nested2' Nested template template_val1 template_val2" # expected
    ),
    (
        """Parent template {parent_kwarg} {parent_func} {nested_template}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"parent_kwarg": "parent_value", "parent_func": some_func, "nested_template": nested_template}, # call_kwargs
        {
            "parent_func": {"kwarg1": "nested1", "kwarg2": "nested2"},
            "nested_template": {"template_kwarg1": "template_val1", "template_kwarg2": "template_val2"}}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Parent template parent_value kwarg1='nested1' kwarg2='nested2' Nested template template_val1 template_val2" # expected
    ),   

])
def test_prompt_template_nested_templates(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template=template, 
                         init_kwargs=init_kwargs,
                         init_nested_kwargs=init_nested_kwargs, 
                         init_template_getter_kwargs=init_template_getter_kwargs, 
                         init_value_formatters=None,
                         call_kwargs=call_kwargs, 
                         call_nested_kwargs=call_nested_kwargs, 
                         call_template_getter_kwargs=call_template_getter_kwargs, 
                         call_value_formatters=None,
                         expected=expected
                         )  
    

@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, expected", [
    (
        get_template, # template
        {"template_kwarg1": "template1_val1"}, # init_kwargs
        {}, # init_nested_kwargs
        {"template_type": "template_type1"}, # init_template_getter_kwargs
        {"template1_kwarg2": "template1_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Template type: template_type1 template1_val1 template1_val2" # expected
    ),
    (
        get_template, # template
        {"template_kwarg1": "template_val2"}, # init_kwargs
        {}, # init_nested_kwargs
        {"template_type": "template_type2"}, # init_template_getter_kwargs
        {"template2_kwarg2": "template2_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        "Template type: template_type2 template_val2 template2_val2" # expected
    ),  
    (
        get_template, # template
        {"template_kwarg1": "template1_val1"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"template1_kwarg2": "template1_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {"template_type": "template_type1"}, # call_template_getter_kwargs
        "Template type: template_type1 template1_val1 template1_val2" # expected
    ),
    (
        get_template, # template
        {"template_kwarg1": "template_val2"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"template2_kwarg2": "template2_val2"}, # call_kwargs
        {}, # call_nested_kwargs
        {"template_type": "template_type2"}, # call_template_getter_kwargs
        "Template type: template_type2 template_val2 template2_val2" # expected
    ),      


])
def test_prompt_template_template_callable(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         expected: str
                         ):
    _test_prompt_template(template=template, 
                         init_kwargs=init_kwargs,
                         init_nested_kwargs=init_nested_kwargs, 
                         init_template_getter_kwargs=init_template_getter_kwargs, 
                         init_value_formatters=None,
                         call_kwargs=call_kwargs, 
                         call_nested_kwargs=call_nested_kwargs, 
                         call_template_getter_kwargs=call_template_getter_kwargs, 
                         call_value_formatters=None,
                         expected=expected
                         )





def string_formatter(value: str):
    return f"<STRING>{value}</STRING>"

def string_formatter2(value: str):
    return f"<STRING2>{value}</STRING2>"

def int_formatter(value: int):
    return f"<INT>{value}</INT>"

def bytes_decode_formatter(value: bytes):
    return value.decode()

def bytes_hex_formatter(value: bytes):
    return value.hex()

def list_newline_formatter(value: list):
    return "\n".join(value)

def list_comma_formatter(value: list):
    return ", ".join(value)

def list_enumerate_formatter(value: list):
    return ", ".join(f"{i+1}. {item}" for i, item in enumerate(value))

def dict_newline_formatter(value: dict):
    return "\n".join(f"{k}: {v}" for k, v in value.items())

def dict_comma_formatter(value: dict):
    return ", ".join(f"{k}: {v}" for k, v in value.items())

def dict_document_title_formatter(value: dict):
    return "\n".join(f"DOCUMENT: {k}:\n{v}" for k, v in value.items())

from pprint import pprint
from io import StringIO
def pprint_to_str(value: Any):
    sio = StringIO()
    pprint(value, stream=sio)
    return sio.getvalue()

def func_returning_str(n: int):
    return f"string{n}"

def func_returning_list(n: int):
    return [i for i in range(n)]

def func_returning_dict(n: int):
    return {f"key{i}": f"value{i}" for i in range(n)}

def func_returning_bytes(n: int):
    return bytes(range(n))

@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, init_value_formatters, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, call_value_formatters, expected", [
    (
        """Test template {str_kwarg}""", # template
        {"str_kwarg": "str_val"}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_kwarg": string_formatter}, # init_value_formatters
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        "Test template <STRING>str_val</STRING>" # expected
    ),
    (
        """Test template {str_kwarg}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_kwarg": string_formatter}, # init_value_formatters
        {"str_kwarg": "str_val"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        "Test template <STRING>str_val</STRING>" # expected
    ),    
    (
        """Test template {str_kwarg} {str_kwarg2} {str_kwarg3}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"str_kwarg": string_formatter, "str_kwarg2": string_formatter2, "str_kwarg3": None}, # init_value_formatters
        {"str_kwarg": "str_val", "str_kwarg2": "str_val2", "str_kwarg3": "str_val3"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        "Test template <STRING>str_val</STRING> <STRING2>str_val2</STRING2> str_val3" # expected
    ),   
    (
        """Test template {str_kwarg} {str_kwarg2} {str_kwarg3}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {str: string_formatter}, # init_value_formatters
        {"str_kwarg": "str_val", "str_kwarg2": "str_val2", "str_kwarg3": "str_val3"}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        "Test template <STRING>str_val</STRING> <STRING>str_val2</STRING> <STRING>str_val3</STRING>" # expected
    ), 
    (
        """Test template {str_kwarg} {bytes_kwarg} {list_kwarg} {dict_kwarg}""", # template
        {}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {
            str: string_formatter,
            bytes: bytes_hex_formatter,
            list: list_comma_formatter,
            dict: dict_comma_formatter
        }, # init_value_formatters
        {
            "str_kwarg": "str_val", 
            "bytes_kwarg": b"bytes_val", 
            "list_kwarg": ["list_val1", "list_val2", "list_val3"],
            "dict_kwarg": {"dict_key1": "dict_val1", "dict_key2": "dict_val2", "dict_key3": "dict_val3"}
        }, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        f'Test template {string_formatter("str_val")} {bytes_hex_formatter(b"bytes_val")} {list_comma_formatter(["list_val1", "list_val2", "list_val3"])} {dict_comma_formatter({"dict_key1": "dict_val1", "dict_key2": "dict_val2", "dict_key3": "dict_val3"})}' # expected
    ),             
])
def test_prompt_template_value_formatters(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         init_value_formatters: Optional[dict],
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         call_value_formatters: Optional[dict],
                         expected: str
                         ):
    _test_prompt_template(template=template, 
                         init_kwargs=init_kwargs,
                         init_nested_kwargs=init_nested_kwargs, 
                         init_template_getter_kwargs=init_template_getter_kwargs, 
                         init_value_formatters=init_value_formatters,
                         call_kwargs=call_kwargs, 
                         call_nested_kwargs=call_nested_kwargs, 
                         call_template_getter_kwargs=call_template_getter_kwargs, 
                         call_value_formatters=call_value_formatters,
                         expected=expected
                         )  

class DayOfWeekGetter:
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    def __init__(self):
        self.day = 0

    def get_current_day(self):
        today = self.days[self.day]
        self.day = (self.day + 1) % len(self.days)
        return today
    

def day_formatter(day: str):
    return f"<TODAY>{day}</TODAY>"


@pytest.mark.parametrize("template, init_kwargs, init_nested_kwargs, init_template_getter_kwargs, init_value_formatters, call_kwargs, call_nested_kwargs, call_template_getter_kwargs, call_value_formatters, expected_list", [
    (
        """Today is {current_day}""", # template
        {"current_day": DayOfWeekGetter().get_current_day}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # init_value_formatters
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        [
            "Today is Monday",
            "Today is Tuesday",
            "Today is Wednesday",
            "Today is Thursday",
            "Today is Friday",
            "Today is Saturday",
            "Today is Sunday",
        ] # expected_list
    ),
    (
        """Today is {current_day}""", # template
        {"current_day": DayOfWeekGetter().get_current_day}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {"current_day": day_formatter}, # init_value_formatters
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        {}, # call_value_formatters
        [
            "Today is <TODAY>Monday</TODAY>",
            "Today is <TODAY>Tuesday</TODAY>",
            "Today is <TODAY>Wednesday</TODAY>",
            "Today is <TODAY>Thursday</TODAY>",
            "Today is <TODAY>Friday</TODAY>",
            "Today is <TODAY>Saturday</TODAY>",
            "Today is <TODAY>Sunday</TODAY>",
        ] # expected_list
    ),    
    (
        """Today is {current_day}""", # template
        {"current_day": DayOfWeekGetter().get_current_day}, # init_kwargs
        {}, # init_nested_kwargs
        {}, # init_template_getter_kwargs
        {}, # init_value_formatters
        {}, # call_kwargs
        {}, # call_nested_kwargs
        {}, # call_template_getter_kwargs
        { "current_day": day_formatter}, # call_value_formatters
        [
            "Today is <TODAY>Monday</TODAY>",
            "Today is <TODAY>Tuesday</TODAY>",
            "Today is <TODAY>Wednesday</TODAY>",
            "Today is <TODAY>Thursday</TODAY>",
            "Today is <TODAY>Friday</TODAY>",
            "Today is <TODAY>Saturday</TODAY>",
            "Today is <TODAY>Sunday</TODAY>",
        ] # expected_list
    ),
])
def test_prompt_template_stateful_callable(template: str|Callable[..., str],
                         init_kwargs: dict, 
                         init_nested_kwargs: Optional[dict], 
                         init_template_getter_kwargs: Optional[dict], 
                         init_value_formatters: Optional[dict],
                         call_kwargs: dict, 
                         call_nested_kwargs: Optional[dict], 
                         call_template_getter_kwargs: Optional[dict], 
                         call_value_formatters: Optional[dict],
                         expected_list: list[str]
                         ):
    
    for expected in expected_list:
        _test_prompt_template(template=template, 
                            init_kwargs=init_kwargs,
                            init_nested_kwargs=init_nested_kwargs, 
                            init_template_getter_kwargs=init_template_getter_kwargs, 
                            init_value_formatters=init_value_formatters,
                            call_kwargs=call_kwargs, 
                            call_nested_kwargs=call_nested_kwargs, 
                            call_template_getter_kwargs=call_template_getter_kwargs, 
                            call_value_formatters=call_value_formatters,
                            expected=expected
                            )      
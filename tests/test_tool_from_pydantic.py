import pytest
from unifai import UnifAIClient, LLMProvider
from unifai.type_conversions import standardize_messages, standardize_tools
from unifai.types import (
    Message, 
    Image, 
    ToolCall, 
    ToolParameter,
    ToolParameters,
    StringToolParameter,
    NumberToolParameter,
    IntegerToolParameter,
    BooleanToolParameter,
    NullToolParameter,
    ObjectToolParameter,
    ArrayToolParameter,
    RefToolParameter,
    AnyOfToolParameter,
    Tool,
    PROVIDER_TOOLS
)

from basetest import base_test_llms_all

from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from typing import Literal, get_args, get_origin, Any, Optional, Union, TypeVar, TypeAlias, Sequence, Mapping, List, Annotated, Union

from unifai.type_conversions.tool_from_pydantic import tool_from_pydantic_model, tool_parameter_from_pydantic, get_field_and_item_origin
ai = UnifAIClient()

class Contact(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    """Name of the contact."""
    
    email: str
    """Email of the contact."""
    
    phone: str
    """Phone number of the contact."""
    
    address: str
    """Address of the contact."""
    
    job_title: str
    """Job title of the contact."""
    
    company: str
    """Company of the contact."""
    
    is_domestic: bool
    """Is the contact domestic?"""
    
    gender: str
    """Gender of the contact."""
    
    confidence: float
    """Confidence score of the contact information."""

class Status(Enum):
    ACTIVE = 'active'
    INACTIVE = 'inactive'


class Address(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    street: str
    """Street name and number."""
    city: str
    """City name."""
    country: Literal['USA', 'UK', 'Canada']
    """Country of the address."""


class User(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    
    favorite_num_or_word: int|str = 1
    """The user's favorite number or word."""
    
    name: str
    """The user's full name."""
    
    age: int = Field(description="The user's age in years.")
    
    status: Status
    """The user's account status. WEINER"""
    
    role: Literal['admin', 'user', 'guest']
    """The user's role in the system."""
    
    address: Address = Field(description="The user's address details (nested model).")

    contacts: list[Contact] = Field(description="List of contacts for the user.")

    favorite_nums: list[int] = [1, 2, 3]
    """List of the user's favorite numbers."""

    favorite_num_or_word: int|str = 1
    """The user's favorite number or word."""

class StringEnum(Enum):
    A = 'a'
    B = 'b'
    C = 'c'

class AnnotatedStringEnum(Enum):
    """String enum field"""
    A = 'a'
    B = 'b'
    C = 'c'

class ModelWithAllDescriptions(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    string_fd: str = Field(description="String field")
    string_il: str = "String field"
    """String field"""
    string_literal_fd: Literal["a", "b", "c"] = Field(description="String literal field")
    string_literal_il: Literal["a", "b", "c"] = "a"
    """String literal field"""
    string_enum_fd: StringEnum = Field(description="String enum field", default=StringEnum.A)
    string_enum_il: StringEnum = StringEnum.A
    """String enum field"""
    string_enum_anno_il: AnnotatedStringEnum = AnnotatedStringEnum.A

    op_string_fd: Optional[str] = Field(description="Optional string field")
    op_string_il: Optional[str] = None
    """Optional string field"""
    op_string_literal_fd: Optional[Literal["a", "b", "c"]] = Field(description="Optional string literal field")
    op_string_literal_il: Optional[Literal["a", "b", "c"]] = None
    """Optional string literal field"""
    op_string_enum_fd: Optional[StringEnum] = Field(description="Optional string enum field", default=StringEnum.A)
    op_string_enum_il: Optional[StringEnum] = StringEnum.A
    """Optional string enum field"""
    op_string_enum_anno_il: Optional[AnnotatedStringEnum] = AnnotatedStringEnum.A
    """Optional string enum field"""

    list_string_fd: list[str] = Field(description="List of string field")
    list_string_il: list[str] = ["a", "b", "c"]
    """List of string field"""
    list_string_literal_fd: list[Literal["a", "b", "c"]] = Field(description="List of string literal field")
    list_string_literal_il: list[Literal["a", "b", "c"]] = ["a", "b", "c"]
    """List of string literal field"""
    list_string_enum_fd: list[StringEnum] = Field(description="List of string enum field", default_factory=list)
    list_string_enum_il: list[StringEnum] = [StringEnum.A, StringEnum.B, StringEnum.C]
    """List of string enum field"""
    list_string_enum_anno_il: list[AnnotatedStringEnum] = [AnnotatedStringEnum.A, AnnotatedStringEnum.B, AnnotatedStringEnum.C]
    """List of string enum field"""

    op_list_string_fd: Optional[list[str]] = Field(description="Optional list of string field")
    op_list_string_il: Optional[list[str]] = None
    """Optional list of string field"""
    op_list_string_literal_fd: Optional[list[Literal["a", "b", "c"]]] = Field(description="Optional list of string literal field")
    op_list_string_literal_il: Optional[list[Literal["a", "b", "c"]]]= None
    """Optional list of string literal field"""
    op_list_string_enum_fd: Optional[list[StringEnum]] = Field(description="Optional list of string enum field", default_factory=list)
    op_list_string_enum_il: Optional[list[StringEnum]] = [StringEnum.A, StringEnum.B, StringEnum.C]
    """Optional list of string enum field"""
    op_list_string_enum_anno_il: Optional[list[AnnotatedStringEnum]] = [AnnotatedStringEnum.A, AnnotatedStringEnum.B, AnnotatedStringEnum.C]
    """Optional list of string enum field"""

class SubModel(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)
    string: str
    integer: int
    number: float
    boolean: bool
    list_string: list[str]


StringAlias = str|bytes
IntAlias = int|bytes
FloatAlias = float|bytes
BoolAlias = bool|bytes
ListStringAlias = list[str]|bytes
ListIntAlias = list[int]|bytes
ListFloatAlias = list[float]|bytes
ListBoolAlias = list[bool]|bytes
ListSubModelAlias = list[SubModel]|bytes

class ModelWithAllAnnoCombos(BaseModel):
    string: str
    integer: int
    number: float
    boolean: bool    
    submodel: SubModel

    op_string: Optional[str]
    op_integer: Optional[int]
    op_number: Optional[float]
    op_boolean: Optional[bool]
    op_submodel: Optional[SubModel]

    list_string: list[str]
    list_integer: list[int]
    list_number: list[float]
    list_boolean: list[bool]
    list_submodel: list[SubModel]

    op_list_string: Optional[list[str]]
    op_list_integer: Optional[list[int]]
    op_list_number: Optional[list[float]]
    op_list_boolean: Optional[list[bool]]
    op_list_submodel: Optional[list[SubModel]]

    list_list_string: list[list[str]]
    list_list_integer: list[list[int]]
    list_list_number: list[list[float]]
    list_list_boolean: list[list[bool]]
    list_list_submodel: list[list[SubModel]]

    op_list_list_string: Optional[list[list[str]]]
    op_list_list_integer: Optional[list[list[int]]]
    op_list_list_number: Optional[list[list[float]]]
    op_list_list_boolean: Optional[list[list[bool]]]
    op_list_list_submodel: Optional[list[list[SubModel]]]

    list_list_list_string: list[list[list[str]]]
    list_list_list_integer: list[list[list[int]]]
    list_list_list_number: list[list[list[float]]]
    list_list_list_boolean: list[list[list[bool]]]
    list_list_list_submodel: list[list[list[SubModel]]]

    anno_string: Annotated[str, Field(description="Annotated string field")]
    anno_integer: Annotated[int, Field(description="Annotated integer field")]
    anno_number: Annotated[float, Field(description="Annotated number field")]
    anno_boolean: Annotated[bool, Field(description="Annotated boolean field")]
    anno_submodel: Annotated[SubModel, Field(description="Annotated submodel field")]

    op_anno_string: Optional[Annotated[str, Field(description="Optional annotated string field")]]
    op_anno_integer: Optional[Annotated[int, Field(description="Optional annotated integer field")]]
    op_anno_number: Optional[Annotated[float, Field(description="Optional annotated number field")]]
    op_anno_boolean: Optional[Annotated[bool, Field(description="Optional annotated boolean field")]]
    op_anno_submodel: Optional[Annotated[SubModel, Field(description="Optional annotated submodel field")]]

    string_alias: StringAlias
    integer_alias: IntAlias
    number_alias: FloatAlias
    boolean_alias: BoolAlias
    list_string_alias: ListStringAlias
    list_integer_alias: ListIntAlias
    list_number_alias: ListFloatAlias
    list_boolean_alias: ListBoolAlias
    list_submodel_alias: ListSubModelAlias
    
    literal_string: Literal["a", "b", "c"]
    literal_integer: Literal[1, 2, 3]
    literal_boolean: Literal[True, False]
    
    op_literal_string: Optional[Literal["a", "b", "c"]]
    op_literal_integer: Optional[Literal[1, 2, 3]]
    op_literal_boolean: Optional[Literal[True, False]]

    op_anno_literal_string: Optional[Annotated[Literal["a", "b", "c"], Field(description="Optional annotated literal string field")]]
    op_anno_literal_integer: Optional[Annotated[Literal[1, 2, 3], Field(description="Optional annotated literal integer field")]]
    op_anno_literal_boolean: Optional[Annotated[Literal[True, False], Field(description="Optional annotated literal boolean field")]]

    list_literal_string: list[Literal["a", "b", "c"]]
    list_literal_integer: list[Literal[1, 2, 3]]
    list_literal_boolean: list[Literal[True, False]]

    op_list_literal_string: Optional[list[Literal["a", "b", "c"]]]
    op_list_literal_integer: Optional[list[Literal[1, 2, 3]]]
    op_list_literal_boolean: Optional[list[Literal[True, False]]]

    list_list_literal_string: list[list[Literal["a", "b", "c"]]]
    list_list_literal_integer: list[list[Literal[1, 2, 3]]]
    list_list_literal_boolean: list[list[Literal[True, False]]]

    enum_string: StringEnum
    op_enum_string: Optional[StringEnum]
    list_enum_string: list[StringEnum]
    op_list_enum_string: Optional[list[StringEnum]]
    list_list_enum_string: list[list[StringEnum]]
    op_list_list_enum_string: Optional[list[list[StringEnum]]]

@pytest.mark.parametrize("model", [User, Address, Contact, ModelWithAllDescriptions, ModelWithAllAnnoCombos])
def test_tool_from_base_model(model):
    param = tool_parameter_from_pydantic(model)
    assert isinstance(param, ObjectToolParameter)
    print(param)
    return_tool = tool_from_pydantic_model(model)
    assert isinstance(return_tool, Tool)
    print(return_tool)



    


# for field_name, field_info in ModelWithAllAnnoCombos.model_fields.items():
#     # print(field_name, field_info)
#     print()
#     print("Name:", field_name)
#     print("Anno:", field_info.annotation)
#     # field_type, item_type = get_field_and_item_origin(field_info.annotation)
#     field_dict = get_field_and_item_origin(field_info.annotation)
#     field_type = field_dict['type']
#     item_type = field_dict['item_type']
#     field_args = field_dict['args']
#     field_enum = field_dict.get('enum')

#     print("FieldType:", field_type)
#     print("ItemType:", item_type)
#     print("FieldArgs:", field_args)
#     print("FieldEnum:", field_enum)
#     # print(field_info.default)
#     # print(field_info.description)


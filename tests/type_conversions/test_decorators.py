import pytest
from unifai import UnifAI, ProviderName
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
    Tool,
)

from basetest import base_test_llms_all

from unifai.type_conversions.tool_from_func import parse_docstring_and_annotations
from unifai.type_conversions import tool
from pydantic import BaseModel, Field

ai = UnifAI()


@pytest.mark.parametrize("docstring, expected_description, expected_parameters", [
    (
        """Get the current weather in a given location

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            unit (str): The unit of temperature to return. Infer the unit from the location if not provided.
                enum: ["celsius", "fahrenheit"]

        Returns:
            dict: The current weather in the location
                condition (str): The weather condition, e.g. "sunny", "cloudy", "rainy", "hot"
                degrees (float): The temperature in the location
                unit (str): The unit of temperature, e.g. "F", "C" 
        """,
        "Get the current weather in a given location",
        ObjectToolParameter(
            type="object",
            properties=[
                StringToolParameter(
                    name="location", 
                    description="The city and state, e.g. San Francisco, CA", 
                ),
                StringToolParameter(
                    name="unit",
                    description="The unit of temperature to return. Infer the unit from the location if not provided.", 
                    enum=["celsius", "fahrenheit"],
                )
            ]
        )
    ),
    (
        """Get the current weather in a given location

        Args:
            location (str): The city and state, 
            e.g. San Francisco, CA

            unit (str): The unit of temperature to return. 
                        Infer the unit from the location if not provided.
                enum: ["celsius", "fahrenheit"]

        Returns:
            dict: The current weather in the location
                condition (str): The weather condition, e.g. "sunny", "cloudy", "rainy", "hot"
                degrees (float): The temperature in the location
                unit (str): The unit of temperature, e.g. "F", "C" 
        """,
        "Get the current weather in a given location",
        ObjectToolParameter(
            type="object",
            properties=[
                StringToolParameter(
                    name="location", 
                    description="The city and state, e.g. San Francisco, CA", 
                ),
                StringToolParameter(
                    name="unit",
                    description="The unit of temperature to return. Infer the unit from the location if not provided.", 
                    enum=["celsius", "fahrenheit"],
                )
            ]
        )
    ),    

    (
        """Get an object with all types

        Args:
            string_param (str): A string parameter
                enum: ["a", "b", "c"]
            number_param (float): A number parameter
                enum: [1.0, 2.0, 3.0]
            integer_param (int): An integer parameter
                enum: [1, 2, 3]
            boolean_param (bool): A boolean parameter
            null_param (None): A null parameter
            array_param (list): An array parameter
                items (str): A string item
                    enum: ["a", "b", "c"]
            object_param (dict): An object parameter
                properties:
                    string_prop (str): A string property
                        enum: ["a", "b", "c"]
                    number_prop (float): A number property
                        enum: [1.0, 2.0, 3.0]
                    integer_prop (int): An integer property
                        enum: [1, 2, 3]
                    boolean_prop (bool): A boolean property
                    null_prop (None): A null property
                    array_prop (list): An array property
                        items (str): A string item
                            enum: ["a", "b", "c"]
                    nested_object_prop (dict): A nested object property
                        properties:
                            nested_string_prop (str): A string property in a nested object
                                enum: ["a", "b", "c"]
                            nested_number_prop (float): A number property in a nested object
                                enum: [1.0, 2.0, 3.0]
                            nested_integer_prop (int): An integer property in a nested object
                                enum: [1, 2, 3]
                            nested_boolean_prop (bool): A boolean property in a nested object
                            nested_null_prop (None): A null property in a nested object
                            nested_array_prop (list): An array property in a nested object
                                items (str): A string item in array in a nested object
                                    enum: ["a", "b", "c"]
            
        """,
        "Get an object with all types",
        ObjectToolParameter(
            properties=[
                StringToolParameter(
                    name="string_param",
                    description="A string parameter",
                    enum=["a", "b", "c"]
                ),
                NumberToolParameter(
                    name="number_param",
                    description="A number parameter",
                    enum=[1.0, 2.0, 3.0]
                ),
                IntegerToolParameter(
                    name="integer_param",
                    description="An integer parameter",
                    enum=[1, 2, 3]
                ),
                BooleanToolParameter(
                    name="boolean_param",
                    description="A boolean parameter",
                ),
                NullToolParameter(
                    name="null_param",
                    description="A null parameter",
                ),
                ArrayToolParameter(
                    name="array_param",
                    description="An array parameter",
                    items=StringToolParameter(
                        description="A string item",
                        enum=["a", "b", "c"]
                    )
                ),
                ObjectToolParameter(
                    name="object_param",
                    description="An object parameter",
                    properties=[
                        StringToolParameter(
                            name="string_prop",
                            description="A string property",
                            enum=["a", "b", "c"]
                        ),
                        NumberToolParameter(
                            name="number_prop",
                            description="A number property",
                            enum=[1.0, 2.0, 3.0]
                        ),
                        IntegerToolParameter(
                            name="integer_prop",
                            description="An integer property",
                            enum=[1, 2, 3]
                        ),
                        BooleanToolParameter(
                            name="boolean_prop",
                            description="A boolean property",
                        ),
                        NullToolParameter(
                            name="null_prop",
                            description="A null property",
                        ),
                        ArrayToolParameter(
                            name="array_prop",
                            description="An array property",
                            items=StringToolParameter(
                                description="A string item",
                                enum=["a", "b", "c"]
                            )
                        ),
                        ObjectToolParameter(
                            name="nested_object_prop",
                            description="A nested object property",
                            properties=[
                                StringToolParameter(
                                    name="nested_string_prop",
                                    description="A string property in a nested object",
                                    enum=["a", "b", "c"]
                                ),
                                NumberToolParameter(
                                    name="nested_number_prop",
                                    description="A number property in a nested object",
                                    enum=[1.0, 2.0, 3.0]
                                ),
                                IntegerToolParameter(
                                    name="nested_integer_prop",
                                    description="An integer property in a nested object",
                                    enum=[1, 2, 3]
                                ),
                                BooleanToolParameter(
                                    name="nested_boolean_prop",
                                    description="A boolean property in a nested object",
                                ),
                                NullToolParameter(
                                    name="nested_null_prop",
                                    description="A null property in a nested object",
                                ),
                                ArrayToolParameter(
                                    name="nested_array_prop",
                                    description="An array property in a nested object",
                                    items=StringToolParameter(
                                        description="A string item in array in a nested object",
                                        enum=["a", "b", "c"]
                                    )
                                ),
                            ],
                        ), # nested_object_prop
                    ],
                ) # object_param
            ],
        ) # ObjectToolParameter
    ),        

])
def test_parse_docstring_and_annotations(docstring, expected_description, expected_parameters):
    description, parameters = parse_docstring_and_annotations(docstring)
    assert description == expected_description
    assert description == expected_description



def test_decorators_get_current_weather():
    
    @tool
    def get_current_weather(location: str, unit: str = "fahrenheit") -> dict:
        """Get the current weather in a given location

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            unit (str): The unit of temperature to return. Infer the unit from the location if not provided.
                enum: ["celsius", "fahrenheit"]

        Returns:
            dict: The current weather in the location
                condition (str): The weather condition, e.g. "sunny", "cloudy", "rainy", "hot"
                degrees (float): The temperature in the location
                unit (str): The unit of temperature, e.g. "F", "C" 
        """

        location = location.lower()
        if 'san francisco' in location:
            degrees = 69
            condition = 'sunny'
        elif 'tokyo' in location:
            degrees = 50
            condition = 'cloudy'
        elif 'paris' in location:
            degrees = 40
            condition = 'rainy'
        else:
            degrees = 100
            condition = 'hot'
        if unit == 'celsius':
            degrees = (degrees - 32) * 5/9
            unit = 'C'
        else:
            unit = 'F'
        return {'condition': condition, 'degrees': degrees, 'unit': unit}
    
    assert get_current_weather("San Francisco, CA") == {'condition': 'sunny', 'degrees': 69, 'unit': 'F'}
    assert get_current_weather("San Francisco, CA", unit="celsius") == {'condition': 'sunny', 'degrees': 20.555555555555557, 'unit': 'C'}        

    assert type(get_current_weather) == Tool
    assert get_current_weather.name == "get_current_weather"
    assert get_current_weather.description == "Get the current weather in a given location"
    assert get_current_weather.parameters == ObjectToolParameter(
        type="object",
        properties=[
            StringToolParameter(
                name="location", 
                description="The city and state, e.g. San Francisco, CA", 
            ),
            StringToolParameter(
                name="unit",
                description="The unit of temperature to return. Infer the unit from the location if not provided.", 
                enum=["celsius", "fahrenheit"],
            )
        ]
    )

def test_decorators_calculator():
    # Test with type annotations in func signature NOT matching the docstring
    
    @tool()
    def calculator(operation: str, left_val: float, right_val: float) -> float:
        """Perform a basic arithmetic operation on two numbers.
        Args:
            operation: The operation to perform.
                enum: ["add", "subtract", "multiply", "divide"]
            left_val: The value on the left side of the operation
            right_val: The value on the right side of the operation
        """
        if operation == "add":
            return left_val + right_val
        elif operation == "subtract":
            return left_val - right_val
        elif operation == "multiply":
            return left_val * right_val
        elif operation == "divide":
            return left_val / right_val
        else:
            return 0
        
    assert calculator("add", 2, 3) == 5
    assert calculator("subtract", 2, 3) == -1
    assert calculator("multiply", 2, 3) == 6
    assert calculator("divide", 6, 3) == 2
    assert type(calculator) == Tool
    assert calculator.name == "calculator"
    assert calculator.description == "Perform a basic arithmetic operation on two numbers."
    assert calculator.parameters == ObjectToolParameter(
        type="object",
        properties=[
            StringToolParameter(
                name="operation",
                description="The operation to perform.",
                enum=["add", "subtract", "multiply", "divide"],
            ),
            NumberToolParameter(
                name="left_val",
                description="The value on the left side of the operation",
            ),
            NumberToolParameter(
                name="right_val",
                description="The value on the right side of the operation",
            )
        ]
    )

def test_decorators_base_model():

    @tool
    class Customer(BaseModel):
        """A customer object"""
        name: str
        age: int
        email: str
        phone: str
        address: str = Field(description="The customer's address")

    assert type(Customer) == Tool
    assert Customer.name == "return_Customer"
    assert Customer.description == "A customer object"
    assert Customer.parameters == ObjectToolParameter(
        type="object",
        properties=[
            StringToolParameter(name="name"),
            IntegerToolParameter(name="age"),
            StringToolParameter(name="email"),
            StringToolParameter(name="phone"),
            StringToolParameter(name="address", description="The customer's address"),
        ]
    )
    customer = Customer(name="John Doe", age=30, email="1", phone="2", address="3")
    assert customer.name == "John Doe"
    assert customer.age == 30
    assert customer.email == "1"
    assert customer.phone == "2"
    assert customer.address == "3"


    @tool(name="create_customer", description="Create a new customer")
    class Customer2(BaseModel):
        """A customer object"""
        name: str
        age: int
        email: str
        phone: str
        address: str

    assert type(Customer2) == Tool
    assert Customer2.name == "create_customer"
    assert Customer2.description == "Create a new customer"
    assert Customer2.parameters == ObjectToolParameter(
        type="object",
        properties=[
            StringToolParameter(name="name"),
            IntegerToolParameter(name="age"),
            StringToolParameter(name="email"),
            StringToolParameter(name="phone"),
            StringToolParameter(name="address"),
        ]
    )

    customer2 = Customer2(name="John Doe", age=30, email="1", phone="2", address="3")
    assert customer2.name == "John Doe"
    assert customer2.age == 30
    assert customer2.email == "1"
    assert customer2.phone == "2"
    assert customer2.address == "3"

    assert Customer2 != Customer
    assert customer2 != customer

class Customer(BaseModel):
    """A customer object"""
    name: str
    age: int
    email: str
    phone: str
    address: str = Field(description="The customer's address")

def test_decorators_base_model_annotations():
    @tool
    def tool_with_basemodel_arg(customer: Customer) -> dict:
        return {"customer": customer.model_dump()}

    assert tool_with_basemodel_arg.name == "tool_with_basemodel_arg"
    assert tool_with_basemodel_arg.description == ''
    assert tool_with_basemodel_arg.parameters == ObjectToolParameter(
        type="object",
        properties=[
            ObjectToolParameter(
                type="object",
                name="customer",
                description="A customer object",
                properties=[
                    StringToolParameter(name="name"),
                    IntegerToolParameter(name="age"),
                    StringToolParameter(name="email"),
                    StringToolParameter(name="phone"),
                    StringToolParameter(name="address", description="The customer's address"),
                ]
            )
        ]
    )
    
    @tool(description="A tool with no docs")
    def tool_with_basemodel_arg2(customer: Customer) -> dict:
        return {"customer": customer.model_dump()}

    assert tool_with_basemodel_arg2.name == "tool_with_basemodel_arg2"
    assert tool_with_basemodel_arg2.description == 'A tool with no docs'
    assert tool_with_basemodel_arg2.parameters == ObjectToolParameter(
        type="object",
        properties=[
            ObjectToolParameter(
                type="object",
                name="customer",
                description="A customer object",
                properties=[
                    StringToolParameter(name="name"),
                    IntegerToolParameter(name="age"),
                    StringToolParameter(name="email"),
                    StringToolParameter(name="phone"),
                    StringToolParameter(name="address", description="The customer's address"),
                ]
            )
        ]
    )

    @tool
    def tool_with_partial_docs(customer: Customer) -> dict:
        """A tool with partial docs"""
        return {"customer": customer.model_dump()}
    
    assert tool_with_partial_docs.name == "tool_with_partial_docs"
    assert tool_with_partial_docs.description == 'A tool with partial docs'
    assert tool_with_partial_docs.parameters == ObjectToolParameter(
        type="object",
        properties=[
            ObjectToolParameter(
                type="object",
                name="customer",
                description="A customer object",
                properties=[
                    StringToolParameter(name="name"),
                    IntegerToolParameter(name="age"),
                    StringToolParameter(name="email"),
                    StringToolParameter(name="phone"),
                    StringToolParameter(name="address", description="The customer's address"),
                ]
            )
        ]
    )
    
    @tool
    def tool_with_description_only_args(int1: int, int2: int) -> int:
        """
        A tool with description only
        Args:
            int1: The first integer
            int2: The second integer
        """
        return int1 + int2
    assert tool_with_description_only_args.name == "tool_with_description_only_args"
    assert tool_with_description_only_args.description == 'A tool with description only'
    assert tool_with_description_only_args.parameters == ObjectToolParameter(
        type="object",
        properties=[
            IntegerToolParameter(name="int1", description="The first integer"),
            IntegerToolParameter(name="int2", description="The second integer"),
        ]
    )

    # @tool
    # def some_tool_with_no_docs(param1: str, param2: int, param3: float, customer: Customer) -> dict:
    #     return {"param1": param1, "param2": param2, "param3": param3, "customer": customer.model_dump()}

    # @tool
    # def tool_with_basemodel_arg2(customer: Customer) -> dict:
    #     return {"customer": customer.model_dump()}

    # print(some_tool_with_no_docs)

    # print(tool_with_basemodel_arg)
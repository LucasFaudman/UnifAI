import pytest
from unifai import UnifAIClient, LLMProvider
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
from pydantic import BaseModel

ai = UnifAIClient()


@pytest.mark.parametrize("docstring, expected_description, expected_parameters", [
    (
        """Get the current weather in a given location

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            unit (str): The unit of temperature to return. Infer the unit from the location if not provided.
                enum: ["celsius", "fahrenheit"]
                required: False

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
                    required=True
                ),
                StringToolParameter(
                    name="unit",
                    description="The unit of temperature to return. Infer the unit from the location if not provided.", 
                    enum=["celsius", "fahrenheit"],
                    required=False
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
                required: False

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
                    required=True
                ),
                StringToolParameter(
                    name="unit",
                    description="The unit of temperature to return. Infer the unit from the location if not provided.", 
                    enum=["celsius", "fahrenheit"],
                    required=False
                )
            ]
        )
    ),    

    (
        """Get an object with all types

        Args:
            string_param (str): A string parameter
                enum: ["a", "b", "c"]
                required: True
            number_param (float): A number parameter
                enum: [1.0, 2.0, 3.0]
                required: True
            integer_param (int): An integer parameter
                enum: [1, 2, 3]
                required: True
            boolean_param (bool): A boolean parameter
                required: True
            null_param (None): A null parameter
                required: True
            array_param (list): An array parameter
                items (str): A string item
                    enum: ["a", "b", "c"]
                required: True
            object_param (dict): An object parameter
                properties:
                    string_prop (str): A string property
                        enum: ["a", "b", "c"]
                        required: True
                    number_prop (float): A number property
                        enum: [1.0, 2.0, 3.0]
                        required: True
                    integer_prop (int): An integer property
                        enum: [1, 2, 3]
                        required: True
                    boolean_prop (bool): A boolean property
                        required: True
                    null_prop (None): A null property
                        required: True
                    array_prop (list): An array property
                        items (str): A string item
                            enum: ["a", "b", "c"]
                        required: True
                    nested_object_prop (dict): A nested object property
                        properties:
                            nested_string_prop (str): A string property in a nested object
                                enum: ["a", "b", "c"]
                                required: True
                            nested_number_prop (float): A number property in a nested object
                                enum: [1.0, 2.0, 3.0]
                                required: True
                            nested_integer_prop (int): An integer property in a nested object
                                enum: [1, 2, 3]
                                required: True
                            nested_boolean_prop (bool): A boolean property in a nested object
                                required: True
                            nested_null_prop (None): A null property in a nested object
                                required: True
                            nested_array_prop (list): An array property in a nested object
                                items (str): A string item in array in a nested object
                                    enum: ["a", "b", "c"]
                                required: True
                        required: True
                required: True
            
        """,
        "Get an object with all types",
        ObjectToolParameter(
            properties=[
                StringToolParameter(
                    name="string_param",
                    description="A string parameter",
                    required=True,
                    enum=["a", "b", "c"]
                ),
                NumberToolParameter(
                    name="number_param",
                    description="A number parameter",
                    required=True,
                    enum=[1.0, 2.0, 3.0]
                ),
                IntegerToolParameter(
                    name="integer_param",
                    description="An integer parameter",
                    required=True,
                    enum=[1, 2, 3]
                ),
                BooleanToolParameter(
                    name="boolean_param",
                    description="A boolean parameter",
                    required=True,
                ),
                NullToolParameter(
                    name="null_param",
                    description="A null parameter",
                    required=True,
                ),
                ArrayToolParameter(
                    name="array_param",
                    description="An array parameter",
                    required=True,
                    items=StringToolParameter(
                        description="A string item",
                        required=True,
                        enum=["a", "b", "c"]
                    )
                ),
                ObjectToolParameter(
                    name="object_param",
                    description="An object parameter",
                    required=True,
                    properties=[
                        StringToolParameter(
                            name="string_prop",
                            description="A string property",
                            required=True,
                            enum=["a", "b", "c"]
                        ),
                        NumberToolParameter(
                            name="number_prop",
                            description="A number property",
                            required=True,
                            enum=[1.0, 2.0, 3.0]
                        ),
                        IntegerToolParameter(
                            name="integer_prop",
                            description="An integer property",
                            required=True,
                            enum=[1, 2, 3]
                        ),
                        BooleanToolParameter(
                            name="boolean_prop",
                            description="A boolean property",
                            required=True,
                        ),
                        NullToolParameter(
                            name="null_prop",
                            description="A null property",
                            required=True,
                        ),
                        ArrayToolParameter(
                            name="array_prop",
                            description="An array property",
                            required=True,
                            items=StringToolParameter(
                                description="A string item",
                                required=True,
                                enum=["a", "b", "c"]
                            )
                        ),
                        ObjectToolParameter(
                            name="nested_object_prop",
                            description="A nested object property",
                            required=True,
                            properties=[
                                StringToolParameter(
                                    name="nested_string_prop",
                                    description="A string property in a nested object",
                                    required=True,
                                    enum=["a", "b", "c"]
                                ),
                                NumberToolParameter(
                                    name="nested_number_prop",
                                    description="A number property in a nested object",
                                    required=True,
                                    enum=[1.0, 2.0, 3.0]
                                ),
                                IntegerToolParameter(
                                    name="nested_integer_prop",
                                    description="An integer property in a nested object",
                                    required=True,
                                    enum=[1, 2, 3]
                                ),
                                BooleanToolParameter(
                                    name="nested_boolean_prop",
                                    description="A boolean property in a nested object",
                                    required=True,
                                ),
                                NullToolParameter(
                                    name="nested_null_prop",
                                    description="A null property in a nested object",
                                    required=True,
                                ),
                                ArrayToolParameter(
                                    name="nested_array_prop",
                                    description="An array property in a nested object",
                                    required=True,
                                    items=StringToolParameter(
                                        description="A string item in array in a nested object",
                                        required=True,
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
    assert expected_description == description
    assert expected_parameters == parameters



def test_decorators_get_current_weather():
    
    @tool
    def get_current_weather(location: str, unit: str = "fahrenheit") -> dict:
        """Get the current weather in a given location

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            unit (str): The unit of temperature to return. Infer the unit from the location if not provided.
                enum: ["celsius", "fahrenheit"]
                required: False

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
                required=True
            ),
            StringToolParameter(
                name="unit",
                description="The unit of temperature to return. Infer the unit from the location if not provided.", 
                enum=["celsius", "fahrenheit"],
                required=False
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
                required=True
            ),
            NumberToolParameter(
                name="left_val",
                description="The value on the left side of the operation",
                required=True
            ),
            NumberToolParameter(
                name="right_val",
                description="The value on the right side of the operation",
                required=True
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
        address: str

    assert type(Customer) == Tool
    assert Customer.name == "return_Customer"
    assert Customer.description == "A customer object"
    assert Customer.parameters == ObjectToolParameter(
        type="object",
        properties=[
            StringToolParameter(name="name", required=True),
            IntegerToolParameter(name="age", required=True),
            StringToolParameter(name="email", required=True),
            StringToolParameter(name="phone", required=True),
            StringToolParameter(name="address", required=True),
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
            StringToolParameter(name="name", required=True),
            IntegerToolParameter(name="age", required=True),
            StringToolParameter(name="email", required=True),
            StringToolParameter(name="phone", required=True),
            StringToolParameter(name="address", required=True),
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

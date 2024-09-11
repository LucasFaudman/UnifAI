import pytest
from unifai import UnifAIClient, AIProvider
from unifai._convert_types import standardize_messages, standardize_tools
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
    FunctionTool,
    CodeInterpreterTool,
    FileSearchTool,
)

from basetest import base_test_all_providers

ai = UnifAIClient()

@pytest.mark.parametrize("input_messages, expected_std_messages", [
    (
        [Message(role='user', content='Hello AI')],
        [Message(role='user', content='Hello AI')]
    ),
    (
        [{'role': 'user', 'content': 'Hello AI'}],
        [Message(role='user', content='Hello AI')]
    ),    
    (
        ['Hello AI'],
        [Message(role='user', content='Hello AI')]
    ),
    (
        [
            Message(role='system', content='Your role is...'), 
            Message(role='user', content='Hello AI'),
            Message(role='assistant', content='Hello User'),
        ],    
        [
            Message(role='system', content='Your role is...'), 
            Message(role='user', content='Hello AI'),
            Message(role='assistant', content='Hello User'),
        ],          
    ),
    (
        [
            {'role': 'system', 'content': 'Your role is...'}, 
            {'role': 'user', 'content': 'Hello AI'},
            {'role': 'assistant', 'content': 'Hello User'},
        ],
        [
            Message(role='system', content='Your role is...'), 
            Message(role='user', content='Hello AI'),
            Message(role='assistant', content='Hello User'),
        ],  
    ),    
    (
        [
            {'role': 'system', 'content': 'Your role is...'}, 
            'Hello AI',
            {'role': 'assistant', 'content': 'Hello User'},
        ],
        [
            Message(role='system', content='Your role is...'), 
            Message(role='user', content='Hello AI'),
            Message(role='assistant', content='Hello User'),
        ],  
    ),
    (
        [
            Message(role='system', content='Your role is...'),
            'Hello AI',
            {'role': 'assistant', 'content': 'Hello User'},
        ],
        [
            Message(role='system', content='Your role is...'), 
            Message(role='user', content='Hello AI'),
            Message(role='assistant', content='Hello User'),
        ],  
    ),
])
def test_standardize_messages(input_messages, expected_std_messages):
    std_messages = standardize_messages(input_messages)
    for std_message, expected_message in zip(std_messages, expected_std_messages):
        # sync created_at before comparison
        std_message.created_at = expected_message.created_at
        assert std_message == expected_message

    assert std_messages == expected_std_messages



TOOL_DICTS = {
    "code_interpreter": {"type": "code_interpreter"},
    "file_search": {"type": "file_search"},
    "get_current_weather": {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            },
            "strict": True
        },
    }, # get_current_weather

    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a basic arithmetic operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "left_value": {
                        "type": "number",
                        "description": "The value on the left side of the operation",
                    },
                    "right_value": {
                        "type": "number",
                        "description": "The value on the right side of the operation",
                    }
                },
                "required": ["operation", "left_value", "right_value"],
                "additionalProperties": False
            },
            "strict": True
        },
    }, # calculator

    "calculator_from_sequence": {
        "type": "function",
        "function": {
            "name": "calculator_from_sequence",
            "description": "Perform a basic arithmetic operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "left_value": {
                        "type": "number",
                        "description": "The value on the left side of the operation",
                    },
                    "right_value": {
                        "type": "number",
                        "description": "The value on the right side of the operation",
                    }
                },
                "required": ["operation", "left_value", "right_value"],
                "additionalProperties": False
            },
            "strict": True
        },
    }, # calculator_from_sequence    

    "calculator_from_mapping": {
        "type": "function",
        "function": {
            "name": "calculator_from_mapping",
            "description": "Perform a basic arithmetic operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "left_value": {
                        "type": "number",
                        "description": "The value on the left side of the operation",
                    },
                    "right_value": {
                        "type": "number",
                        "description": "The value on the right side of the operation",
                    }
                },
                "required": ["operation", "left_value", "right_value"],
                "additionalProperties": False
            },
            "strict": True
        },
    }, # calculator_from_mapping 

    "calculator_from_args": {
        "type": "function",
        "function": {
            "name": "calculator_from_args",
            "description": "Perform a basic arithmetic operation on two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "The operation to perform",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    },
                    "left_value": {
                        "type": "number",
                        "description": "The value on the left side of the operation",
                    },
                    "right_value": {
                        "type": "number",
                        "description": "The value on the right side of the operation",
                    }
                },
                "required": ["operation", "left_value", "right_value"],
                "additionalProperties": False
            },
            "strict": True
        },
    }, # calculator_from_args

    "get_object_with_all_types": {
        "type": "function",
        "function": {
            "name": "get_object_with_all_types",
            "description": "Get an object with all types",
            "parameters": {
                "type": "object",
                "properties": {
                    "string_param": {
                        "type": "string",
                        "description": "A string parameter",
                        "enum": ["a", "b", "c"]
                    },
                    "number_param": {
                        "type": "number",
                        "description": "A number parameter",
                        "enum": [1.0, 2.0, 3.0]
                    },
                    "integer_param": {
                        "type": "integer",
                        "description": "An integer parameter",
                        "enum": [1, 2, 3]
                    },
                    "boolean_param": {
                        "type": "boolean",
                        "description": "A boolean parameter",
                        # "enum": [True, False]
                    },
                    "null_param": {
                        "type": "null",
                        "description": "A null parameter",
                        # "enum": [None]
                    },
                    "array_param": {
                        "type": "array",
                        "description": "An array parameter",
                        "items": {
                            "type": "string",
                            "description": "A string item",
                            "enum": ["a", "b", "c"]                            
                        }
                    },
                    "object_param": {
                        "type": "object",
                        "description": "An object parameter",
                        "properties": {
                            "string_prop": {
                                "type": "string",
                                "description": "A string property",
                                "enum": ["a", "b", "c"]
                            },
                            "number_prop": {
                                "type": "number",
                                "description": "A number property",
                                "enum": [1.0, 2.0, 3.0]
                            },
                            "integer_prop": {
                                "type": "integer",
                                "description": "An integer property",
                                "enum": [1, 2, 3]
                            },
                            "boolean_prop": {
                                "type": "boolean",
                                "description": "A boolean property",
                                # "enum": [True, False]
                            },
                            "null_prop": {
                                "type": "null",
                                "description": "A null property",
                                # "enum": [None]
                            },
                            "array_prop": {
                                "type": "array",
                                "description": "An array property",
                                "items": {
                                    "type": "string",
                                    "description": "A string item",
                                    "enum": ["a", "b", "c"]                            
                                }
                            },
                            "nested_object_prop": {
                                "type": "object",
                                "description": "A nested object property",
                                "properties": {
                                    "nested_string_prop": {
                                        "type": "string",
                                        "description": "A string property in a nested object",
                                        "enum": ["a", "b", "c"]
                                    },
                                    "nested_number_prop": {
                                        "type": "number",
                                        "description": "A number property in a nested object",
                                        "enum": [1.0, 2.0, 3.0]
                                    },
                                    "nested_integer_prop": {
                                        "type": "integer",
                                        "description": "An integer property in a nested object",
                                        "enum": [1, 2, 3]
                                    },
                                    "nested_boolean_prop": {
                                        "type": "boolean",
                                        "description": "A boolean property in a nested object",
                                        # "enum": [True, False]
                                    },
                                    "nested_null_prop": {
                                        "type": "null",
                                        "description": "A null property in a nested object",
                                        # "enum": [None]
                                    },
                                    "nested_array_prop": {
                                        "type": "array",
                                        "description": "An array property in a nested object",
                                        "items": {
                                            "type": "string",
                                            "description": "A string item in array in a nested object",
                                            "enum": ["a", "b", "c"]                            
                                        }
                                    },
                                },
                                "required": ["nested_string_prop", "nested_number_prop", "nested_integer_prop", "nested_boolean_prop", "nested_null_prop", "nested_array_prop"],
                                "additionalProperties": False
                            },
                        },
                        "required": ["string_prop", "number_prop", "integer_prop", "boolean_prop", "null_prop", "array_prop", "nested_object_prop"],
                        "additionalProperties": False
                    }
                },
                "required": ["string_param", "number_param", "integer_param", "boolean_param", "null_param", "array_param", "object_param"],
                "additionalProperties": False
            },
            "strict": True
        }
    }, # get_object_with_all_types
}    

TOOL_OBJECTS = {
    "code_interpreter": CodeInterpreterTool(),
    "file_search": FileSearchTool(),

    "get_current_weather": FunctionTool(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters=ObjectToolParameter(
            # name="parameters",
            properties=[
                StringToolParameter(
                    name="location",
                    description="The city and state, e.g. San Francisco, CA",
                    required=True
                ),
                StringToolParameter(
                    name="unit",
                    enum=["celsius", "fahrenheit"],
                    required=False
                )
            ],
        )
    ), # get_current_weather

    "calculator": FunctionTool(
        name="calculator",
        description="Perform a basic arithmetic operation on two numbers.",
        parameters=ObjectToolParameter(
            properties=[
                StringToolParameter(
                    name="operation",
                    description="The operation to perform",
                    required=True,
                    enum=["add", "subtract", "multiply", "divide"]
                ),
                NumberToolParameter(
                    name="left_value",
                    description="The value on the left side of the operation",
                    required=True
                ),
                NumberToolParameter(
                    name="right_value",
                    description="The value on the right side of the operation",
                    required=True
                ),
            ]
        )
    ), # calculator

    "calculator_from_sequence": FunctionTool(
        name="calculator_from_sequence",
        description="Perform a basic arithmetic operation on two numbers.",
        parameters=[
            StringToolParameter(
                name="operation",
                description="The operation to perform",
                required=True,
                enum=["add", "subtract", "multiply", "divide"]
            ),
            NumberToolParameter(
                name="left_value",
                description="The value on the left side of the operation",
                required=True
            ),
            NumberToolParameter(
                name="right_value",
                description="The value on the right side of the operation",
                required=True
            ),
        ]
    ), # calculator_from_sequence    

    "calculator_from_mapping": FunctionTool(
        name="calculator_from_mapping",
        description="Perform a basic arithmetic operation on two numbers.",
        parameters={
            "operation": StringToolParameter(
                description="The operation to perform",
                required=True,
                enum=["add", "subtract", "multiply", "divide"]
            ),
            "left_value": NumberToolParameter(
                description="The value on the left side of the operation",
                required=True,
            ),
            "right_value": NumberToolParameter(
                description="The value on the right side of the operation",
                required=True
            ),
        }
    ), # calculator_from_mapping

    "calculator_from_args": FunctionTool(
        "calculator_from_args",
        "Perform a basic arithmetic operation on two numbers.",
        StringToolParameter(
            name="operation",
            description="The operation to perform",
            enum=["add", "subtract", "multiply", "divide"],
            required=True
        ),
        NumberToolParameter(
            name="left_value",
            description="The value on the left side of the operation",
            required=True
        ),
        NumberToolParameter(
            name="right_value",
            description="The value on the right side of the operation",
            required=True
        ),
    ), # calculator_from_args    

    "get_object_with_all_types": FunctionTool(
        name="get_object_with_all_types",
        description="Get an object with all types",
        parameters=ObjectToolParameter(
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
        )
    )
            
}

@pytest.mark.parametrize("input_tools, expected_std_tools", [
    (
        [TOOL_DICTS["code_interpreter"]],
        [TOOL_OBJECTS["code_interpreter"]]
    ), 
    (
        [TOOL_DICTS["file_search"]],
        [TOOL_OBJECTS["file_search"]],
    ),     
    (
        [TOOL_DICTS["get_current_weather"]],
        [TOOL_OBJECTS["get_current_weather"]]
    ),    
    (
        [TOOL_DICTS["get_object_with_all_types"]],
        [TOOL_OBJECTS["get_object_with_all_types"]]
    ),
    (
        # test all tools
        list(TOOL_DICTS.values()),
        list(TOOL_OBJECTS.values())
    )
    
])
def test_standardize_tools(input_tools, expected_std_tools):
    std_tools = list(standardize_tools(input_tools).values())
    dict_tools = [tool.to_dict() for tool in std_tools]

    # assert len(std_tools) == len(expected_std_tools)
    # for std_tool, expected_tool in zip(std_tools, expected_std_tools):
    #     for std_param, expected_param in zip(std_tool.parameters.properties, expected_tool.parameters.properties):
    #         assert std_param == expected_param

    #     assert std_tool.name == expected_tool.name
    #     assert std_tool.description == expected_tool.description
    #     assert std_tool.parameters.required == expected_tool.parameters.required
    #     assert std_tool == expected_tool

    assert std_tools == expected_std_tools
    assert dict_tools == input_tools

    
import pytest
from simplifai import SimplifAI, AIProvider
from simplifai._types import Message, Tool
from basetest import base_test_all_providers

@base_test_all_providers
def test_chat_simple(provider: AIProvider, client_kwargs: dict, func_kwargs: dict):

    ai = SimplifAI({provider: client_kwargs})
    ai.init_client(provider, **client_kwargs)
    messages = ai.chat(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        provider=provider,
        **func_kwargs
    )
    assert messages
    assert isinstance(messages, list)

    for message in messages:
        assert isinstance(message, Message)
        assert message.content
        print(f'{message.role}: {message.content}')
        if message.tool_calls:
            for tool_call in message.tool_calls:
                print(f'Tool Call: {tool_call.tool_name}')
                print(tool_call.arguments)
    print()

messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]

tools = [
        {
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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
]

def get_current_weather(location: str, unit: str = "fahrenheit") -> dict:
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
    return {'degrees': degrees, 'unit': unit, 'condition': condition}


@base_test_all_providers
@pytest.mark.parametrize("messages, tools, tool_callables", [
    (
        [
            {"role": "system", "content": "Your role is use the available tools to answer questions like a cartoon Pirate"},            
            {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"},
        ],
        [
        {
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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location", "unit"],
                },
            },
        }
        ],
        {
            "get_current_weather": get_current_weather
        }
    ),
                        ])

def test_chat_tools_simple(
    # ai: SimplifAI, 
    provider: AIProvider, 
    client_kwargs: dict,
    func_kwargs: dict,
    messages: list,
    tools: list,
    tool_callables: dict
    ):

    ai = SimplifAI(
        {provider: client_kwargs},
        tool_callables=tool_callables
    )
    ai.init_client(provider, **client_kwargs)
    messages = ai.chat(
        messages=messages,
        provider=provider,
        tools=tools,
        **func_kwargs
    )
    assert messages
    assert isinstance(messages, list)

    for message in messages:
        print(f'{message.role}:\n{message.content or message.tool_calls}\n')
        assert isinstance(message, Message)
        assert message.content or message.tool_calls
    print()
    
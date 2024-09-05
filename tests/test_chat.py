import pytest
from unifai import UnifAIClient, AIProvider
from unifai._types import Message, Tool
from basetest import base_test_all_providers

# TOOLS AND TOOL CALLABLES
TOOLS = {
    "get_current_weather": 
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
                    "unit": {
                        "type": "string", 
                        "description": "The unit of temperature to return. Infer the unit from the location if not provided.",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location", "unit"],
            },
        }
    },
    "return_weather_messages":
    {
        "type": "function",
        "function": {
            "name": "return_weather_messages",
            "description": "Return a message about the current weather for one or more locations",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "The messages to return about the weather",
                        "items":{
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "The message for the weather in the location",
                                },
                            },
                            "required": ["location", "message"],
                        },
                    },
                },
                "required": ["messages"],
            }
        }
    },

}

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
    return {'condition': condition, 'degrees': degrees, 'unit': unit}












@base_test_all_providers
def test_chat_simple(provider: AIProvider, client_kwargs: dict, func_kwargs: dict):

    ai = UnifAIClient({provider: client_kwargs})
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



@base_test_all_providers
@pytest.mark.parametrize("messages, tools, tool_callables", [
    (
        [
            {"role": "system", "content": "Your role is use the available tools to answer questions like a cartoon Pirate"},            
            {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris? Respond in Fahrenheit."},
        ],
        [
            TOOLS["get_current_weather"]
        ],
        {
            "get_current_weather": get_current_weather
        }
    ),

])
def test_chat_tools_simple(
    # ai: UnifAIClient, 
    provider: AIProvider, 
    client_kwargs: dict,
    func_kwargs: dict,
    messages: list,
    tools: list,
    tool_callables: dict
    ):

    ai = UnifAIClient(
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
    


@base_test_all_providers
@pytest.mark.parametrize("messages, tools, tool_callables", [
    (
        [
            {"role": "system", "content": "Your role is use the available tools to answer questions like a cartoon Pirate"},            
            {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris? Respond in Fahrenheit."},
        ],
        [
            TOOLS["get_current_weather"]
        ],
        {
            "get_current_weather": get_current_weather
        }
    ),
])
def test_chat_return_on(
    # ai: UnifAIClient, 
    provider: AIProvider, 
    client_kwargs: dict,
    func_kwargs: dict,
    messages: list,
    tools: list,
    tool_callables: dict, 
    ):

    ai = UnifAIClient(
        {provider: client_kwargs},
        tool_callables=tool_callables
    )
    ai.init_client(provider, **client_kwargs)

    return_ons = ["content"]
    if tool_names := [tool["function"]["name"] for tool in tools]:
        return_ons.append("message")
        return_ons.append("tool_call")
        return_ons.append(tool_names[0])
        return_ons.append(tool_names)
    

    for return_on in return_ons:
        new_messages = ai.chat(
            messages=messages,
            provider=provider,
            tools=tools,
            return_on=return_on,
            **func_kwargs
        )

        assert new_messages
        last_message = new_messages[-1]
        assert isinstance(last_message, Message)

        if return_on == "content":
            assert last_message.content
            assert not last_message.tool_calls

        elif return_on == "message":
            assert last_message.content or last_message.tool_calls

        elif return_on == "tool_call":
            assert last_message.tool_calls
            # assert not last_message.content

        elif tool_names and return_on == tool_names[0]:
            assert last_message.tool_calls
            # assert not last_message.content
            assert last_message.tool_calls[0].tool_name == tool_names[0]

        elif tool_names and return_on == tool_names:
            assert last_message.tool_calls
            # assert not last_message.content
            assert last_message.tool_calls[0].tool_name in tool_names            

        for message in new_messages:
            print(f'\n{message.role}:\n{message.content or message.tool_calls}\n')            
        print()




@base_test_all_providers
@pytest.mark.parametrize("messages, tools, tool_callables, tool_choice", [
    (
        [
            {"role": "system", "content": "Your role is use the available tools to answer questions like a cartoon Pirate"},            
            {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris? Respond in Fahrenheit."},
        ],
        [
            TOOLS["get_current_weather"]
        ],
        {
            "get_current_weather": get_current_weather
        }, 
        "get_current_weather"
    ),
])
def test_chat_enforce_tool_choice(
    # ai: UnifAIClient, 
    provider: AIProvider, 
    client_kwargs: dict,
    func_kwargs: dict,
    messages: list,
    tools: list,
    tool_callables: dict, 
    tool_choice: str
    ):

    ai = UnifAIClient(
        {provider: client_kwargs},
        tool_callables=tool_callables
    )
    ai.init_client(provider, **client_kwargs)

    for _ in range(1):
        new_messages = ai.chat(
            messages=messages,
            provider=provider,
            tools=tools,
            tool_choice=tool_choice,
            return_on="message",
            enforce_tool_choice=True,
            **func_kwargs
        )
        
        assert new_messages
        assert isinstance(new_messages, list)

        last_message = new_messages[-1]
        assert isinstance(last_message, Message)

        if tool_choice == 'auto':
            assert last_message.content or last_message.tool_calls
        elif tool_choice == 'required':
            assert last_message.tool_calls
            assert not last_message.content
        elif tool_choice == 'none':
            assert last_message.content
            assert not last_message.tool_calls
        else:
            assert last_message.tool_calls
            called_tools = [tool_call.tool_name for tool_call in last_message.tool_calls]
            assert tool_choice in called_tools


        for message in new_messages:
            print(f'\n{message.role}:\n{message.content or message.tool_calls}\n')            
    print()


@base_test_all_providers
@pytest.mark.parametrize("messages, tools, tool_callables, tool_choice", [
    (
        [
            {"role": "system", "content": "Your role is use the available tools to answer questions like a cartoon Pirate"},            
            {"role": "user", "content": "What's the weather like in San Francisco? Respond in Fahrenheit."},
        ],
        [
            TOOLS["get_current_weather"], TOOLS["return_weather_messages"]
        ],
        {
            "get_current_weather": get_current_weather
        }, 
        ["get_current_weather", "return_weather_messages"]
    ),
])
def test_chat_enforce_tool_choice_sequence(
    # ai: UnifAIClient, 
    provider: AIProvider, 
    client_kwargs: dict,
    func_kwargs: dict,
    messages: list,
    tools: list,
    tool_callables: dict, 
    tool_choice: list[str]
    ):

    ai = UnifAIClient(
        {provider: client_kwargs},
        tool_callables=tool_callables
    )
    ai.init_client(provider, **client_kwargs)

    for _ in range(1):
        new_messages = ai.chat(
            messages=messages,
            provider=provider,
            tools=tools,
            tool_choice=tool_choice,
            return_on=tool_choice[-1],
            enforce_tool_choice=True,
            **func_kwargs
        )
        
        assert new_messages
        assert isinstance(new_messages, list)

        last_message = new_messages[-1]
        assert isinstance(last_message, Message)

        if tool_choice == 'auto':
            assert last_message.content or last_message.tool_calls
        elif tool_choice == 'required':
            assert last_message.tool_calls
            assert not last_message.content
        elif tool_choice == 'none':
            assert last_message.content
            assert not last_message.tool_calls
        else:
            assert last_message.tool_calls
            called_tools = [tool_call.tool_name for tool_call in last_message.tool_calls]
            assert tool_choice[-1] in called_tools


        for message in new_messages:
            print(f'\n{message.role}:\n{message.content or message.tool_calls}\n')            
    print()
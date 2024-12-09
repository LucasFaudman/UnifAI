import pytest
from unifai import UnifAI, ProviderName, tool
from unifai.types import Message, Tool, MessageChunk
from basetest import base_test_llms_all, PROVIDER_DEFAULTS



@base_test_llms_all
def test_chat_stream_simple(provider: ProviderName, client_kwargs: dict, func_kwargs: dict):

    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])
    stream = ai.chat_stream(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        provider=provider,
        **func_kwargs
    )
    for message_chunk in stream:
        assert isinstance(message_chunk, MessageChunk)
        assert isinstance(message_chunk.content, str)
        print(message_chunk.content, flush=True, end="")
    print("\n")



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


@base_test_llms_all
def test_chat_stream_tools(provider: ProviderName, client_kwargs: dict, func_kwargs: dict):

    ai = UnifAI(provider_configs=[{"provider": provider, "client_init_kwargs": client_kwargs}])
    stream = ai.chat_stream(
        messages=[{"role": "user", "content": "What's the weather in San Francisco, Tokyo, and Paris?"}],
        tools=[get_current_weather],
        tool_choice=["get_current_weather", "auto"],
        # enforce_tool_choice=True,
        provider=provider,
        **func_kwargs
    )
    for message_chunk in stream:
        assert isinstance(message_chunk, MessageChunk)
        if message_chunk.content:
            print(message_chunk.content, flush=True, end="")
        if message_chunk.tool_calls:
            for tool_call in message_chunk.tool_calls:
                print(f"\nCalled Tool: {tool_call.tool_name} with args: {tool_call.arguments}")
    print("\n")
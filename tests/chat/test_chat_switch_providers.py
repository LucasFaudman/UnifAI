import pytest
from unifai import UnifAI, ProviderName, tool, MessageChunk
from unifai.types import Message, Tool
from basetest import base_test_llms, API_KEYS

@pytest.mark.parametrize("provider1", [
    "anthropic",
    "google",
    "openai",
    "ollama",
    "nvidia",
])
@pytest.mark.parametrize("provider2", [
    "anthropic",
    "google",
    "openai",
    "ollama",
    "nvidia",
])
def test_switch_providers_simple(
    provider1: ProviderName, 
    provider2: ProviderName, 
):
    ai = UnifAI(api_keys=API_KEYS)
    chat = ai.chat([Message(role="user", content="Hi my favorite color is blue, what's yours?")], llm=provider1)
    assert isinstance(chat.messages, list)
    assert isinstance(chat.last_content, str)
    print(chat.last_content)
    
    chat.llm_provider = provider2
    assert chat.llm_provider == provider2
    ass_message = chat.send_message(Message(role="user", content="What berry is my favorite color?"))
    assert ass_message.role == "assistant"
    assert ass_message.content
    assert "blueberr" in ass_message.content.lower()
    print(ass_message)



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


@pytest.mark.parametrize("provider1", [
    "anthropic",
    "google",
    "openai",
    "ollama",
    "nvidia",
])
@pytest.mark.parametrize("provider2", [
    "anthropic",
    "google",
    "openai",
    "ollama",
    "nvidia",
])
def test_switch_providers_tool_calls(
    provider1: ProviderName, 
    provider2: ProviderName, 
):
    ai = UnifAI(api_keys=API_KEYS)
    chat = ai.start_chat(
        llm=provider1, 
        tools=[get_current_weather],
        tool_choice=[get_current_weather, "none", "none"],
        enforce_tool_choice=True,        
        )
    message = Message(role="user", content="What's the weather today in San Francisco, Tokyo, and Paris?")
    for message_chunk in chat.send_message_stream(message):
        assert isinstance(message_chunk, MessageChunk)
        if message_chunk.content:
            print(message_chunk.content, flush=True, end="")
        if message_chunk.tool_calls:
            for tool_call in message_chunk.tool_calls:
                print(f"\nCalled Tool: {tool_call.tool_name} with args: {tool_call.arguments}")

    assert isinstance(chat.messages, list)
    assert isinstance(chat.last_content, str)
    print(f"Switch Providers: {provider1} -> {provider2}")
    chat.llm = provider2
    assert chat.llm_provider == provider2
    
    ass_message = chat.send_message(Message(role="user", content="Is it sunny in San Francisco today?. Use previous tool calls to answer this."))
    assert ass_message.role == "assistant"
    assert ass_message.content
    assert "sunny" in ass_message.content.lower()
    # print(ass_message)


    # for message in chat.messages:
    #     print(f"{message.role}: {message.tool_calls or message.content}")
    #     print()
    
    num_user_messages = sum(1 for message in chat.messages if message.role == "user")
    num_ass_messages = sum(1 for message in chat.messages if message.role == "assistant")
    num_tool_messages = sum(1 for message in chat.messages if message.role == "tool")
    print("User Messages:", num_user_messages)
    print("Assistant Messages:", num_ass_messages)
    print("Tool Messages:", num_tool_messages)        

    assert num_user_messages == 2 
    assert num_ass_messages == 3
    assert num_tool_messages == 1
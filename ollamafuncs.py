import requests
from ollama import Client as OllamaClient
from ollama import Client as OllamaClient
from ollama._types import (
    ChatResponse as OllamaChatResponse,
    Message as OllamaMessage,
    Tool as OllamaTool, 
    ToolFunction as OllamaToolFunction,
    ToolCallFunction as OllamaToolCallFunction,
    ToolCall as OllamaToolCall, 
    Parameters as OllamaParameters, 
    Property as OllamaProperty,
    Options as OllamaOptions
)



def dict_to_ollama_tool(tool_dict: dict) -> OllamaTool:
    tool_type = tool_dict['type']
    tool_def = tool_dict[tool_type]

    tool_name = tool_def['name']
    tool_description = tool_def['description']
    tool_parameters = tool_def['parameters']
    parameters_type = tool_parameters['type']
    parameters_required = tool_parameters['required']

    properties = {}
    for prop_name, prop_def in tool_parameters['properties'].items():
        prop_type = prop_def['type']
        prop_description = prop_def.get('description', None)
        prop_enum = prop_def.get('enum', None)
        properties[prop_name] = OllamaProperty(type=prop_type, description=prop_description, enum=prop_enum)
    
    parameters = OllamaParameters(type=parameters_type, required=parameters_required, properties=properties)
    tool_function = OllamaToolFunction(name=tool_name, description=tool_description, parameters=parameters)
    return OllamaTool(type=tool_type, function=tool_function)

def dict_to_ollama_message(message_dict: dict) -> OllamaMessage:
    role = message_dict['role']
    content = message_dict['content']
    images = message_dict.get('images', None)
    tool_calls = message_dict.get('tool_calls', None)
    return OllamaMessage(role=role, content=content, images=images, tool_calls=tool_calls)


client = OllamaClient("http://librem-2.local:11434")
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


# stream = client.chat(
#     model="mistral:7b-instruct",
#     messages=[dict_to_ollama_message(message) for message in messages],
#     tools=[dict_to_ollama_tool(tool) for tool in tools],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)


# from json import load as json_load, dump as json_dump
# from pprint import pprint
# from collections import defaultdict

# with open('geeda.json') as f:
#     geeda = json_load(f)

# geeda_embeddings = defaultdict(list)
# for chapter_title, chapter_text in geeda.items():
#     for line in chapter_text.splitlines():
#         print("geeda Text: ", line)
#         embeddings = client.embed(
#             model="mistral:7b-instruct",
#             input=[line],
#         )
#         geeda_embeddings[chapter_title].append(embeddings)
#         print("Embeddings: ", embeddings)
#         print()

# with open('geeda_embeddings.json', 'w') as f:
#     json_dump(geeda_embeddings, f) 

# pprint(geeda_embeddings)

# embeddings = client.embed(
#     model="mistral:7b-instruct",
#     input=["Hello, world!"],
# )

# print(embeddings)

# print('\nresp:\n', resp)
# print('\ncontent:\n',resp['message']['content'])
# print('\ntool_calls:\n', '\n'.join(map(str, resp['message']['tool_calls'])))


# class OllamaFunctionCaller:



    
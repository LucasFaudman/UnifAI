import pytest
from unifai import UnifAIClient, LLMProvider
from unifai.types import Message, Tool, Image, StringToolParameter
from basetest import base_test_llms_all, LLM_PROVIDERS, base_test

from pathlib import Path
resouces_path = Path(__file__).parent / "resources"

TEST_IMAGES = {
    "dog": {
        "jpeg": {
            "path": str(resouces_path / "dog.jpeg"),
            "url": "https://www.southwindvets.com/files/southeast-memphis-vet-best-small-dog-breed-for-families.jpeg"
        },
        "jpg": {
            "path": str(resouces_path / "dog.jpg"),
            "url": "https://hips.hearstapps.com/hmg-prod/images/chihuahua-dog-running-across-grass-royalty-free-image-1580743445.jpg"
        },
        "png": {
            "path": str(resouces_path / "dog.png"),
            "url": "https://www.wellnesspetfood.com/wp-content/uploads/2024/01/BODY_Small-Dogs_Photo-Credit-Joe-Caione.png"
        },
        "webp": {
            "path": str(resouces_path / "dog.webp"),
            "url": "https://www.petrescue.org.nz/wp-content/uploads/2023/12/Small-Dog-Breed-in-NZ-Havanese.webp"
        } 
    },

}

from base64 import b64encode
for image_name, image_formats in TEST_IMAGES.items():
    for image_format, image_data in image_formats.items():
        with open(image_data["path"], "rb") as f:
            base64_bytes = b64encode(f.read())
            base64_str = base64_bytes.decode("utf-8")
            data_uri = f"data:image/{image_format};base64,{base64_str}"

            TEST_IMAGES[image_name][image_format]["base64_bytes"] = base64_bytes
            TEST_IMAGES[image_name][image_format]["base64_str"] = base64_str
            TEST_IMAGES[image_name][image_format]["data_uri"] = data_uri








@base_test(*LLM_PROVIDERS, exclude=["openai"])
@pytest.mark.parametrize("image_source" , [
    "base64_bytes",
    "base64_str",    
    "path",
    "data_uri",
    # "url"
])
@pytest.mark.parametrize("image_format" , [
    # "jpeg", 
    "jpg", 
    # "png", 
    # "webp"
])
@pytest.mark.parametrize("image_name" , [
    "dog"
])
def test_image_input_animals(
    provider: LLMProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    image_source: str,
    image_format: str,
    image_name: str,    
    ):

    if provider == "openai":
        func_kwargs["model"] = "gpt-4-turbo"
    if provider == "ollama":
        func_kwargs["model"] = "llava-llama3:latest"        

    if image_source.startswith("base64"):
        image = Image.from_base64(
            base64_data=TEST_IMAGES[image_name][image_format][image_source],
            mime_type=f"image/{image_format}"
        )
    elif image_source == "path":
        image = Image.from_file(path=TEST_IMAGES[image_name][image_format]["path"])
    elif image_source == "data_uri":
        image = Image.from_data_uri(data_uri=TEST_IMAGES[image_name][image_format]["data_uri"])        
    elif image_source == "url":
        image = Image.from_url(url=TEST_IMAGES[image_name][image_format]["url"])

    
    print(f"Image Source: {image_source}")
    print(f"Image Format: {image_format}")
    print(f"Image Name: {image_name}")
    print(str(image)[:100])

    messages = [
        Message(role="user", 
                content="Explain what animal is in the image.",
                images=[image]
        ),
    ]


    ai = UnifAIClient({provider: client_kwargs})
    ai.init_client(provider, **client_kwargs)
    chat = ai.chat(
        messages=messages,
        **func_kwargs
    )    
    assert chat.last_content
    assert image_name in chat.last_content.lower()

    messages = chat.messages
    assert messages
    assert isinstance(messages, list)

    for message in messages:
        assert isinstance(message, Message)
        assert message.content or message.images or message.tool_calls
        print(f'{message.role}: {message.content or message.images or message.tool_calls}')

        if message.role == "assistant":
            assert message.response_info
            assert isinstance(message.response_info.model, str)
            assert message.response_info.done_reason == "stop"
            usage = message.response_info.usage
            assert usage
            assert isinstance(usage.input_tokens, int)
            assert isinstance(usage.output_tokens, int)
            assert usage.total_tokens == usage.input_tokens + usage.output_tokens


    print()



@base_test(*LLM_PROVIDERS, exclude=["ollama"])
@pytest.mark.parametrize("image_source" , [
    "base64_bytes",
    "base64_str",    
    "path",
    "data_uri",
    # "url"
])
@pytest.mark.parametrize("image_format" , [
    # "jpeg", 
    "jpg", 
    # "png", 
    # "webp"
])
@pytest.mark.parametrize("image_name" , [
    "dog"
])
def test_image_and_tools_input_animals(
    provider: LLMProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    image_source: str,
    image_format: str,
    image_name: str,    
    ):

    if provider == "openai":
        func_kwargs["model"] = "gpt-4o"
    if provider == "ollama":
        func_kwargs["model"] = "llava-llama3:latest"        

    if image_source.startswith("base64"):
        image = Image.from_base64(
            base64_data=TEST_IMAGES[image_name][image_format][image_source],
            mime_type=f"image/{image_format}"
        )
    elif image_source == "path":
        image = Image.from_file(path=TEST_IMAGES[image_name][image_format]["path"])
    elif image_source == "data_uri":
        image = Image.from_data_uri(data_uri=TEST_IMAGES[image_name][image_format]["data_uri"])        
    elif image_source == "url":
        image = Image.from_url(url=TEST_IMAGES[image_name][image_format]["url"])

    
    print(f"Image Source: {image_source}")
    print(f"Image Format: {image_format}")
    print(f"Image Name: {image_name}")
    print(str(image)[:100])

    messages = [
        Message(role="user", 
                # content="Explain what animal is in the image.",
                images=[image]
        ),
    ]

    return_animal_in_image = Tool(
    name="return_animal_in_image",
    description="Return the animal in the image",
    parameters=[
        StringToolParameter(
            name="animal",
            description="The animal in the image. Ie. cat, bird, dog, etc.",
        ),
        StringToolParameter(
            name="physical_description",
            description="The physical description of the animal",
        )
    ]
)


    ai = UnifAIClient({provider: client_kwargs})
    ai.init_client(provider, **client_kwargs)
    chat = ai.chat(
        messages=messages,
        tools=[return_animal_in_image],
        tool_choice="return_animal_in_image",
        return_on="tool_call",
        **func_kwargs
    )    
    # assert chat.last_content
    # assert image_name in chat.last_content.lower()

    assert chat.last_tool_call
    assert chat.last_tool_call.tool_name == "return_animal_in_image"
    assert chat.last_tool_call.arguments == chat.last_tool_call_args
    last_args = chat.last_tool_call_args
    assert last_args
    assert isinstance(last_args, dict)
    assert "animal" in last_args and last_args["animal"]
    assert "physical_description" in last_args and last_args["physical_description"]
    
    assert image_name in last_args["animal"].lower() or image_name in last_args["physical_description"].lower()
    print(f"Animal: {last_args['animal']}")
    print(f"Physical Description: {last_args['physical_description']}")

    messages = chat.messages
    assert messages
    assert isinstance(messages, list)

    for message in messages:
        assert isinstance(message, Message)
        assert message.content or message.images or message.tool_calls
        print(f'{message.role}: {message.content or message.images or message.tool_calls}')

        if message.role == "assistant":
            assert message.response_info
            assert isinstance(message.response_info.model, str)
            assert message.response_info.done_reason == "tool_calls"
            usage = message.response_info.usage
            assert usage
            assert isinstance(usage.input_tokens, int)
            assert isinstance(usage.output_tokens, int)
            assert usage.total_tokens == usage.input_tokens + usage.output_tokens


    print()



# @base_test_all_providers
# @pytest.mark.parametrize("messages, expected_words_in_content", [
#     # Image from URL
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromUrl(url=DOG_IMAGES["jpeg"]["url"])]
#             ),
#         ],
#         ["dog"]      
#     ),
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromUrl(url=DOG_IMAGES["jpg"]["url"])]
#             ),
#         ],
#         ["dog"]      
#     ),
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromUrl(url=DOG_IMAGES["png"]["url"])]
#             ),
#         ],
#         ["dog"]      
#     ),
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromUrl(url=DOG_IMAGES["webp"]["url"])]
#             ),
#         ],
#         ["dog"]      
#     ),   

#     # Image from File
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromFile(path=DOG_IMAGES["jpeg"]["path"])]
#             ),
#         ],
#         ["dog"]      
#     ),
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromFile(path=DOG_IMAGES["jpg"]["path"])]
#             ),
#         ],
#         ["dog"]      
#     ),
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromFile(path=DOG_IMAGES["png"]["path"])]
#             ),
#         ],
#         ["dog"]      
#     ),
#     (
#         [
#             Message(role="user", 
#                     content="Explain what animal is in the image.",
#                     images=[ImageFromFile(path=DOG_IMAGES["webp"]["path"])]
#             ),
#         ],
#         ["dog"]      
#     ),       


# ])
# def test_image_input(
#     provider: AIProvider, 
#     client_kwargs: dict, 
#     func_kwargs: dict,
#     messages: list,
#     expected_words_in_content: list[str]
#     ):

#     ai = UnifAIClient({provider: client_kwargs})
#     ai.init_client(provider, **client_kwargs)
#     chat = ai.chat(
#         messages=[{"role": "user", "content": "Hello, how are you?"}],
#         provider=provider,
#         **func_kwargs
#     )
#     messages = chat.messages
#     assert messages
#     assert isinstance(messages, list)

#     for message in messages:
#         assert isinstance(message, Message)
#         assert message.content
#         print(f'{message.role}: {message.content}')

#         if message.role == "assistant":
#             assert message.response_info
#             assert isinstance(message.response_info.model, str)
#             assert message.response_info.done_reason == "stop"
#             usage = message.response_info.usage
#             assert usage
#             assert isinstance(usage.input_tokens, int)
#             assert isinstance(usage.output_tokens, int)
#             assert usage.total_tokens == usage.input_tokens + usage.output_tokens


#     print()
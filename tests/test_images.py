import pytest
from unifai import UnifAIClient, AIProvider
from unifai.types import Message, Tool, Image, ImageFromBase64, ImageFromFile, ImageFromDataURI, ImageFromUrl
from basetest import base_test_all_providers

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








@base_test_all_providers
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
    provider: AIProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    image_source: str,
    image_format: str,
    image_name: str,    
    ):

    if provider == "openai":
        func_kwargs["model"] = "gpt-4-0125-preview"

    if image_source.startswith("base64"):
        image = ImageFromBase64(
            base64_data=TEST_IMAGES[image_name][image_format][image_source],
            media_type=f"image/{image_format}"
        )
    elif image_source == "path":
        image = ImageFromFile(path=TEST_IMAGES[image_name][image_format]["path"])
    elif image_source == "data_uri":
        image = ImageFromDataURI(data_uri=TEST_IMAGES[image_name][image_format]["data_uri"])        
    elif image_source == "url":
        image = ImageFromUrl(url=TEST_IMAGES[image_name][image_format]["url"])

    
    print(f"Image Source: {image_source}")
    print(f"Image Format: {image_format}")
    print(f"Image Name: {image_name}")
    print(image)

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
        provider=provider,
        **func_kwargs
    )    
    assert chat.last_content
    assert image_name in chat.last_content.lower()

    messages = chat.messages

    assert messages
    assert isinstance(messages, list)

    for message in messages:
        assert isinstance(message, Message)
        assert message.content
        print(f'{message.role}: {message.content}')

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
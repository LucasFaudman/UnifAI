from unifai import UnifAI, tool, Message

from _provider_defaults import API_KEYS

ai = UnifAI(api_keys=API_KEYS)

messages = ["Hello this is a test"]

for provider in ["nvidia", "openai", "google", "anthropic", "ollama"]:
    print(f"Provider: {provider}\nModel: {ai._get_llm(provider).default_model}\n>>>")
    for message_chunk in ai.chat_stream(messages=messages, provider=provider):
        print(message_chunk.content, flush=True, end="")
    print("\n")


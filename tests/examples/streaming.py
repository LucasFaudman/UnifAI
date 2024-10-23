from unifai import UnifAIClient, tool, Message

from _provider_defaults import PROVIDER_DEFAULTS


ai = UnifAIClient(
    provider_client_kwargs={
        "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],
        "nvidia": PROVIDER_DEFAULTS["nvidia"][1],
        "cohere": PROVIDER_DEFAULTS["cohere"][1],
    }
)
messages = ["Hello this is a test"]

for provider in ["nvidia", "openai", "google", "anthropic", #"ollama"
                #  "google", "anthropic", "ollama"
                 ]:
    # try:
    print(f"Provider: {provider}\nModel: {ai.get_component(provider).default_model}\n>>>")
    for message_chunk in ai.chat_stream(messages=messages, provider=provider):
        print(message_chunk.content, flush=True, end="")
    print("\n")
    # except Exception as e:
    #     print(e)
    #     print()
    #     continue
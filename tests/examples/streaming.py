from unifai import UnifAIClient, tool, Message

from _provider_defaults import PROVIDER_DEFAULTS


ai = UnifAIClient(
    provider_client_kwargs={
        "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1]
    }
)
messages = ["Hello this is a test"]

for provider in ["openai", "google", "anthropic", "ollama"]:
    try:
        for chunk in ai.chat(messages=messages, provider=provider, stream=True):
            print(chunk)
            print()
    except Exception as e:
        print(e)
        print()
        continue
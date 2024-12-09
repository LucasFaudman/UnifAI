from unifai import UnifAI, tool, Message

from _provider_defaults import PROVIDER_DEFAULTS


ai = UnifAI(
    provider_init_kwargs={
        "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],
        "nvidia": PROVIDER_DEFAULTS["nvidia"][1],
        "cohere": PROVIDER_DEFAULTS["cohere"][1],
    }
)


provider1 = input("Enter provider 1 (default: nvidia): ") or "nvidia"
provider2 = input("Enter provider 2 (default: openai): ") or "openai"

system_prompt1 = input("Enter system prompt 1 (default: None): ") or None
system_prompt2 = input("Enter system prompt 2 (default: None): ") or None

default_first_message = "What is the meaning of life?"
first_message = input(f"Enter first message (default: {default_first_message}): ") or default_first_message
num_messages = int(input("Enter number of messages (default: 5): ") or 5)


chat1 = ai.start_chat(provider=provider1, system_prompt=system_prompt1, messages=[Message(role="assistant", content=first_message)])
chat2 = ai.start_chat(provider=provider2, system_prompt=system_prompt2)
print(f"Provider 1: {provider1}\nModel 1: {chat1.model}\nSystem Prompt 1: {chat1.system_prompt}\n")
print(f"Provider 2: {provider2}\nModel 2: {chat2.model}\nSystem Prompt 2: {chat2.system_prompt}\n")

print(f"\033[31;1mMESSAGE: {0}\033[0m")
print(f"\033[35;1m{provider1}: {first_message}\033[0m")
print(f"\033[35;1m{provider2}: ")
for chunk in chat2.send_message_stream(first_message):
    print(chunk.content, end="", flush=True)
print('\n\033[0m')

message_count = 1
while message_count <= num_messages:
    print(f"\033[31;1mMESSAGE: {message_count}\033[0m")

    if message_count % 2 == 0:
        print(f"\033[35;1m{provider1}: ")
        for chunk in chat1.send_message_stream(chat2.last_message.content):
            print(chunk.content, end="", flush=True)
    else:
        print(f"\033[32;1m{provider2}: ")
        for chunk in chat2.send_message_stream(chat1.last_message.content):
            print(chunk.content, end="", flush=True)
    print('\n\033[0m')
    message_count += 1




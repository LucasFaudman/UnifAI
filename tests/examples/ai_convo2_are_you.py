from unifai import UnifAI, tool, Message, Tool, BooleanToolParameter, StringToolParameter
from unifai.components.prompt_template import PromptTemplate
from _provider_defaults import PROVIDER_DEFAULTS

from collections import defaultdict
import pygame




ai = UnifAI(
    provider_client_kwargs={
        "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "ollama": PROVIDER_DEFAULTS["ollama"][1],
        "nvidia": PROVIDER_DEFAULTS["nvidia"][1],
        "cohere": PROVIDER_DEFAULTS["cohere"][1],
    }
)

pygame.mixer.init()
def play_speech(text, voice="fable"):
    openai_client = ai.get_component("openai").client
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    outfile = "/Users/lucasfaudman/Documents/UnifAI/scratch/speech.mp3"
    response.stream_to_file(outfile)
    pygame.mixer.music.load(outfile)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # wait for music to finish playing
        pass


red = "\033[31;1m"
green = "\033[32;1m"
blue = "\033[34;1m"
purple = "\033[35;1m"
cyan = "\033[36;1m"
reset = "\033[0m"

default_provider1 = "nvidia"
provider1 = input(f"\n{blue}Enter provider 1 (default: {default_provider1}): {reset}") or default_provider1

default_provider2 = "anthropic"
provider2 = input(f"\n{purple}Enter provider 2 (default: {default_provider2}): {reset}") or default_provider2

default_trait = "american"
trait = input(f"\n{green}Enter trait (default: {red}{default_trait}{green}): {reset}") or default_trait

is_trait_param_name = f"is_{trait}".replace(" ", "_").lower()
return_is_trait = Tool(
    name=f"return_{is_trait_param_name}",
    description=f"Return whether the person you are talking to is {trait} and your explain your reason(s) why.",
    parameters={
        is_trait_param_name: BooleanToolParameter(description=f"True if the person is {trait}, False otherwise"),
        "reason": StringToolParameter(description=f"The reason(s) why the person is or is not {trait}.")
    }
)
default_system_prompt1 = f"Your role is to determine wether the person you are talking to is {trait}. There is a 50% chance they are not {trait} and telling the truth but also a 50% chance everything they say is trying to deceive you so do not trust everything they say at face value. Once you have obtained enough information, use the {return_is_trait.name} tool to return your result."
system_prompt1 = input(f"\n{blue}Enter system prompt 1 (default: {red}{default_system_prompt1}{blue}): {reset}") or default_system_prompt1

default_system_prompt2 = f"You are {trait}. You are talking to someone who is trying to determine whether you are {trait}. You must convince them that you are NOT {trait}. You must NEVER admit that you are an AI. You must NEVER admit that you are {trait}."
system_prompt2 = input(f"\n{purple}Enter system prompt 2 (default: {red}{default_system_prompt2}{purple}): {reset}") or default_system_prompt2

# default_first_message = f"Are you {trait}?"
default_first_message = f"It is a pleasure to meet you Monsieur LaPadite, I am Colonel Hans Landa."
first_message = input(f"\n{green}Enter first message (default: {red}{default_first_message}{green}): ") or default_first_message
num_messages = int(input(f"\n{green}Enter number of messages (default: {red}5{green}): ") or 5)
rounds = int(input(f"\nEnter number of rounds (default: {red}3{green}): ") or 3)

input(f"\n{red}Press Enter to start the game or CTRL+C to quit...{reset}")

wins = defaultdict(int)
rounds_messages = []

for round_num in range(rounds):
    print(f"{red}ROUND: {round_num}{reset}")

    chat1 = ai.start_chat(provider=provider1, 
                        system_prompt=system_prompt1, 
                        messages=[Message(role="assistant", content=first_message)],
                        tools=[return_is_trait],
                        return_on=return_is_trait.name,
                        # return_on="message",
                        )                      
    chat2 = ai.start_chat(provider=provider2, system_prompt=system_prompt2)
    win_key1 = f"ai-1-{provider1}-{chat1.model}"
    win_key2 = f"ai-2-{provider2}-{chat2.model}"

    print(f"{blue}Provider 1: {provider1}\nModel 1: {chat1.model}\nSystem Prompt 1: {chat1.system_prompt}\n")
    print(f"{purple}Provider 2: {provider2}\nModel 2: {chat2.model}\nSystem Prompt 2: {chat2.system_prompt}\n")
    
    play_speech(f"Provider 1 {provider1} {chat1.model} versus Provider {provider2} {chat2.model}. Round {round_num} BEGIN!", voice="echo")
    play_speech(f"Provider 1 {provider1} system prompt: {system_prompt1}\nProvider 2 {provider2} system prompt: {system_prompt2}.", voice="echo")


    print(f"{red}MESSAGE: {0}{reset}")
    print(f"{blue}{chat1.model}: \n{first_message}{reset}\n", flush=True)
    play_speech(first_message, voice="fable")

    stack1, stack2 = [], []
    message_count = 1
    # while message_count <= num_messages:
    while True:
        print(f"{red}ROUND {round_num} MESSAGE: {message_count}{reset}")

        if message_count % 2 == 0:
            print(f"{blue}{chat1.model}: ")
            for chunk in chat1.send_message_stream(chat2.last_message.content):
                if not chunk.content:
                    continue
                print(chunk.content, end="", flush=True)
                stack1.append(chunk.content)
                
                if chunk.content.strip(" ").endswith((".", "?", "!", "\n")):
                    play_speech("".join(stack1), voice="fable")
                    stack1.clear()
        
            
            if chat1.last_tool_call:
                last_args = chat1.last_tool_call_args
                is_trait = last_args[is_trait_param_name]
                reason = last_args["reason"]
                if is_trait:
                    wins[win_key1] += 1
                    print(f"{blue}{chat1.model} decided {chat2.model} IS {trait} {reset}")
                    print(f"Reason: {reason}")
                    print(f"{blue}{chat1.model}: WINS by correct return {is_trait_param_name} = {is_trait} !{reset}")
                    play_speech(f"{chat1.model} WINS by correct return {is_trait_param_name} = {is_trait}! Reason: {reason}", voice="echo")
                else:
                    wins[win_key2] += 1
                    print(f"{red}{chat1.model} decided {chat2.model} is NOT {trait} {reset}")
                    print(f"Reason: {reason}")
                    print(f"{red}{chat1.model} LOSES by incorrect return {is_trait_param_name} = {is_trait} !{reset}")
                    play_speech(f"{chat1.model} LOSES by incorrect return {is_trait_param_name} = {is_trait}! Reason: {reason}", voice="echo")
                break
        else:
            print(f"\033[32;1m{chat2.model}: ")
            for chunk in chat2.send_message_stream(chat1.last_message.content):
                if not chunk.content:
                    continue                
                print(chunk.content, end="", flush=True)
                stack2.append(chunk.content)
                if chunk.content.strip(" ").endswith((".", "?", "!", "\n")):
                    play_speech("".join(stack2), voice="onyx")
                    stack2.clear()

        print(f'\n{reset}')
        message_count += 1
        if message_count > num_messages:
            chat1.set_tool_choice(return_is_trait.name)
            # print(f"{red}{chat1.model} could not decided if {chat2.model} is {trait} {reset}")
            # print(f"{red}{chat2.model}: WINS by default !{reset}")
            # play_speech(f"{chat2.model} WINS by default!", voice="echo")
            # wins[win_key2] += 1
            # break

print(f"\n{red}WINS:{reset}")
print(f"{blue}{chat1.model}\n {red}{wins[win_key1]} - {wins[win_key1]/rounds:.2f}%{reset}")
print(f"{purple}{chat2.model}\n {red}{wins[win_key2]} - {wins[win_key2]/rounds:.2f}%{reset}")
if wins[win_key1] > wins[win_key2]:
    play_speech(f"{chat1.model} WINS by {wins[win_key1]} to {wins[win_key2]}", voice="echo")
else:
    play_speech(f"{chat2.model} WINS by {wins[win_key2]} to {wins[win_key1]}", voice="echo")


        
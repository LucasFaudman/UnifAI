from unifai import UnifAIClient, tool

from _provider_defaults import PROVIDER_DEFAULTS

class SimpleChat:

    def __init__(self):
        self.ai = UnifAIClient(
            provider_client_kwargs={
                "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
                "google": PROVIDER_DEFAULTS["google"][1],
                "openai": PROVIDER_DEFAULTS["openai"][1],
                "ollama": PROVIDER_DEFAULTS["ollama"][1]
            }
        )

        self.chats = [self.ai.chat()]
        self.current_chat = self.chats[0]

    def run(self):
        while True:
            try: 
                user_message = input(f"{self.header()}\nEnter a message or Ctrl+C to change options\n>>> ")
                print(f"User: '{user_message}'")
                print("Thinking...", end="\r")
                assistant_message = self.current_chat.send_message(user_message)
                print(f"Assistant: {assistant_message.content or assistant_message.tool_calls}")
            except KeyboardInterrupt:
                self.select_option()


    def header(self):
        return f"""
Total Chats: {len(self.chats)}
Current chat: {self.current_chat}
Current provider: {self.current_chat.provider}
Current model: {self.current_chat.model}
Current system prompt: {self.current_chat.system_prompt}
Current tool choice: {self.current_chat.std_tool_choice}
Current response format: {self.current_chat.std_response_format}
        """


    def select_option(self):
        menu =  f"""        
{self.header()}

Current Chat Options:
1. Set Provider
2. Set Model
3. Set System Prompt
4. Set Tool Choice
5. Set Response Format
6. Clear Chat


Switch Chats:
7. New chat
8. Switch chat

Select an option or Ctrl+C to exit: """

        match input(menu).strip():
            case "1":
                self.set_provider()
            case "2":
                self.set_model()
            case "3":
                self.set_system_prompt()
            case "4":
                self.set_tool_choice()
            case "5":
                self.set_response_format()
            case "6":
                self.clear_chat()
            case "7":
                self.new_chat()
            case "8":
                self.switch_chat()


    def set_provider(self):
        if not self.current_chat:
            print("No chat selected")
            return
        
        menu = """
Providers:
1. anthropic
2. google
3. openai
4. ollama

Select a provider: """

        match input(menu).strip().lower():
            case "1":
                self.current_chat.set_provider("anthropic")
            case "2":
                self.current_chat.set_provider("google")
            case "3":
                self.current_chat.set_provider("openai")
            case "4":
                self.current_chat.set_provider("ollama")
            case _:
                print("Invalid provider")
                return


    def set_model(self):
        if not self.current_chat:
            print("No chat selected")
            return
        
        menu = """Models:\n"""
        models = dict(enumerate(self.ai.list_models(provider=self.current_chat.provider), start=1))
        menu += '\n'.join([f"{i}. {model}" for i, model in models.items()])
        model_index = int(input(menu).strip())
        model = models[model_index]
        self.current_chat.set_model(model)

    def set_system_prompt(self):
        if not self.current_chat:
            print("No chat selected")
            return        
        menu = f"""
        Current system prompt: {self.current_chat.system_prompt}
        Enter a new system prompt or leave blank to clear:
        """
        system_prompt = input(menu).strip()
        self.current_chat.set_system_prompt(system_prompt)
        

    def set_tool_choice(self):
        if not self.current_chat:
            print("No chat selected")
            return
        
        menu = """
Select a tool choice:
1. auto
2. required
3. none
        """
        match input(menu).strip():
            case "1":
                self.current_chat.set_tool_choice("auto")
            case "2":
                self.current_chat.set_tool_choice("required")
            case "3":
                self.current_chat.set_tool_choice("none")
            case _:
                print("Invalid tool choice")
                return

    def set_response_format(self):
        menu = """
        Select a response format:
        1. text
        2. json
        """
        if not self.current_chat:
            print("No chat selected")
            return
        
        match input(menu).strip():
            case "1":
                self.current_chat.set_response_format("text")
            case "2":
                self.current_chat.set_response_format("json")
            case _:
                print("Invalid response format")
                return
        

    def clear_chat(self):
        if self.current_chat:
            self.current_chat.clear_messages()
        print("Chat cleared")

    def new_chat(self):
        self.current_chat = self.ai.chat()
        self.chats.append(self.current_chat)
        
    def switch_chat(self):
        menu = f"""
        Select a chat to switch to:        
        """ + '\n'.join([f"{i+1}. {chat.last_message}" for i, chat in enumerate(self.chats)])
        chat_index = int(input(menu).strip()) - 1
        self.current_chat = self.chats[chat_index]


if __name__ == "__main__":
    SimpleChat().run()
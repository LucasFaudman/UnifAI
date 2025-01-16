from .._base_components._base_chat import BaseChat, ChatConfig

class Chat(BaseChat[ChatConfig]):
    component_type = "chat"
    provider = "default"
    config_class = ChatConfig
from typing import Optional, Union, Sequence, Any, Literal, Mapping

from .baseaiclientwrapper import BaseAIClientWrapper


class AnthropicWrapper(BaseAIClientWrapper):

    def import_client(self):
        from anthropic import Anthropic
        return Anthropic
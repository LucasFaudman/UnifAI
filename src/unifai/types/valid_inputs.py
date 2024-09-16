from typing import Any, Literal, Union
from .eval_parameters import EvaluateParameters
from .message import Message
from .tool import Tool

# Supported AI providers
AIProvider = Literal["anthropic", "openai", "ollama"]

# Valid input types that can be converted to a EvaluateParameters object
EvaluateParametersInput = Union[EvaluateParameters, dict[str, Any]]

# Valid input types that can be converted to a Message object
MessageInput = Union[Message,  dict[str, Any], str]

# Valid input types that can be converted to a Tool object
ToolInput = Union[Tool, dict[str, Any], str]
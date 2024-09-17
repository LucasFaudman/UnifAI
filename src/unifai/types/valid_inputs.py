from typing import Any, Literal, Union, Sequence
from .eval_parameters import EvaluateParameters
from .message import Message
from .tool import Tool

# Supported AI providers
AIProvider = Literal["anthropic", "google", "openai", "ollama"]

# Valid input types that can be converted to a Message object
MessageInput = Union[Message,  dict[str, Any], str]

# Valid input types that can be converted to a Tool object
ToolInput = Union[Tool, dict[str, Any], str]

# Valid input types that can be converted to a ToolChoice object
ToolChoiceInput = Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]
# ToolChoiceInput = Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence["ToolChoiceInput"]]

# Valid input types that can be converted to a ResponseFormat object
ResponseFormatInput = Union[Literal["text", "json", "json_schema"], dict[Literal["type"], Literal["text", "json", "json_schema"]]]

# Valid input types that can be converted to a EvaluateParameters object
EvaluateParametersInput = Union[EvaluateParameters, dict[str, Any]]
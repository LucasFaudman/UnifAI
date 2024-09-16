from typing import Any, Mapping, Sequence, Union
from json import dumps as json_dumps


def make_content_serializeable(content: Any) -> Union[str, int, float, bool, dict, list, None]:
    """Recursively makes an object serializeable by converting it to a dict or list of dicts and converting all non-string values to strings."""
    if content is None or isinstance(content, (str, int, float, bool)):
        return content
    if isinstance(content, Mapping):
        return {k: make_content_serializeable(v) for k, v in content.items()}
    if isinstance(content, Sequence):
        return [make_content_serializeable(item) for item in content]
    return str(content) 


def stringify_content(content: Any) -> str:
    """Formats content for use a message content. If content is not a string, it is converted to a json string."""
    if isinstance(content, str):
        return content
    return json_dumps(make_content_serializeable(content), separators=(',', ':'))
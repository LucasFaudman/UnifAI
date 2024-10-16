from typing import Any, Mapping, Collection, Union, Callable
from json import dumps as json_dumps


def make_content_serializeable(content: Any, default: Callable[[Any], str]=str) -> Union[str, int, float, bool, dict, list, None]:
    """Recursively makes an object serializeable by converting it to a dict or list of dicts and converting all non-string values to strings."""
    if content is None or isinstance(content, (str, int, float, bool)):
        return content
    if isinstance(content, Mapping):
        return {k: make_content_serializeable(v) for k, v in content.items()}
    if isinstance(content, Collection):
        return [make_content_serializeable(item) for item in content]
    return default(content)


def stringify_content(content: Any, separators: tuple[str, str]=(',', ':')) -> str:
    """Formats content for use a message content. If content is not a string, it is converted to a json string."""
    if isinstance(content, str):
        return content
    if isinstance(content, memoryview):
        return content.tobytes().decode()
    if isinstance(content, bytes):
        return content.decode()
    return json_dumps(make_content_serializeable(content), separators=separators)
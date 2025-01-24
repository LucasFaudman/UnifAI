from typing import Any, Callable, Union, Mapping, Collection, Literal, Optional, Pattern
from json import dumps as json_dumps
from re import compile as re_compile

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

def clean_text(
    text: str,
    replacements: Optional[dict[str|Pattern, str]] = {
            r'.\x08': '', # Remove backspace formatting
            r'[\x00-\x08\x0B-\x1F\x7F-\x9F]+': ' ', # Replace common control chars with space
    },    
    strip_chars: Optional[str|Literal[False]] = None,
) -> str:
    if not text:
        return text

    if replacements is not None:
        for pattern, replacement in replacements.items():
            if not isinstance(pattern, Pattern):
                pattern = re_compile(pattern)
            text = pattern.sub(replacement, text)

    if strip_chars is not False:
        text = text.strip(strip_chars)
    
    return text

from random import choices as random_choices
from string import ascii_letters, digits

ASCII_LETTERS_AND_DIGITS = ascii_letters + digits

def generate_random_id(length: int = 8, chars: str = ASCII_LETTERS_AND_DIGITS):
    return ''.join(random_choices(chars, k=length))

def sha256_hash(content: Any, string_func: Callable[[Any], str] = str) -> str:
    """Hashes a string using SHA-256."""
    from hashlib import sha256
    _bytes = string_func(content).encode() if not isinstance(content, bytes) else content
    return sha256(_bytes).hexdigest()
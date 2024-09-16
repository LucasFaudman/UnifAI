from typing import Any, Literal, Optional, Sequence, Union
from unifai.types import Message
from .stringify_content import stringify_content

def make_few_shot_prompt(        
        system_prompt: Optional[str] = None, 
        examples: Optional[list[Union[Message, dict[Literal["input", "response"], Any]]]] = None, 
        content: Any = ""
    ) -> Sequence[Message]:
    """Makes list of message objects from system prompt, examples, and user input."""
    messages = []
    if system_prompt:
        # Begin with system_prompt if it exists
        messages.append(Message(role="system", content=system_prompt))

    # Add examples
    if examples:
        for example in examples:
            if isinstance(example, Message):
                messages.append(example)
            else:
                messages.append(Message(role="user", content=stringify_content(example['input'])))
                messages.append(Message(role="assistant", content=stringify_content(example['response'])))

    # Add content
    messages.append(Message(role="user", content=stringify_content(content)))
    return messages
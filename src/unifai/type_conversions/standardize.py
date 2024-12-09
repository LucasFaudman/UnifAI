from typing import Any, Iterable, Union, Optional, Literal, Type, TypeVar
from pydantic import BaseModel
from ..types import (
    
    Message,
    MessageInput,
    Tool,
    ToolInput,
    ToolChoice,
    ResponseFormatInput,
)
from .tools import tool_from_dict, tool_from_func, tool_from_pydantic

T = TypeVar("T")

def standardize_message(message: MessageInput) -> Message:
    if isinstance(message, Message):
        return message
    if isinstance(message, str):
        return Message(role="user", content=message)
    if isinstance(message, dict):
        return Message(**message)
    raise ValueError(f"Invalid message type: {type(message)}")


def standardize_messages(messages: Iterable[MessageInput]) -> list[Message]:
    return [standardize_message(message) for message in messages]


def standardize_tool(tool: ToolInput, tool_dict: Optional[dict[str, Tool]] = None) -> Tool:
    if isinstance(tool, Tool):
        return tool
    elif isinstance(tool, BaseModel):
        return tool_from_pydantic(tool)
    elif callable(tool):
        return tool_from_func(tool)
    elif isinstance(tool, dict):
        return tool_from_dict(tool)                        
    elif isinstance(tool, str):
        if tool_dict and (std_tool := tool_dict.get(tool)):
            return std_tool
        else:
            raise ValueError(f"Tool '{tool}' not found in tools")
    else:    
        raise ValueError(f"Invalid tool type: {type(tool)}") 


def standardize_tools(tools: Iterable[ToolInput], tool_dict: Optional[dict[str, Tool]] = None) -> dict[str, Tool]:
    return {tool.name: tool for tool in (standardize_tool(tool, tool_dict) for tool in tools)}


# def standardize_tool_choice(tool_choice: ToolChoiceInput) -> str|list[str]:
#     if isinstance(tool_choice, str):
#         return tool_choice if tool_choice != 'any' else 'required'
#     if isinstance(tool_choice, Tool):
#         return tool_choice.name
#     if isinstance(tool_choice, dict):
#         tool_type = tool_choice['type']
#         return tool_choice[tool_type]['name']
#     if isinstance(tool_choice, Iterable):
#         tool_choice_str_sequence = []
#         for tool_choice_item in tool_choice:
#             if not isinstance(tool_choice_item, str) and isinstance(tool_choice_item, Iterable):
#                 raise ValueError(f"Invalid tool_choice_item type: {type(tool_choice_item)}. Nested sequences are NOT supported.")            
#             tool_choice_str_sequence.append(standardize_tool_choice(tool_choice_item))
                
#         return tool_choice_str_sequence

#     raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}")

def standardize_tool_choice(tool_choice: ToolChoice) -> str:
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice != 'any' else 'required'
    if isinstance(tool_choice, Tool):
        return tool_choice.name
    if isinstance(tool_choice, dict):
        tool_type = tool_choice['type']
        return tool_choice[tool_type]['name']
    
    raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}")


def standardize_response_format(response_format: ResponseFormatInput) -> str:
    if isinstance(response_format, str):
        return response_format
    if isinstance(response_format, dict):
        return response_format['json_schema']
    raise ValueError(f"Invalid response_format type: {type(response_format)}")

def standardize_config(config: T|dict, config_type: Type[T]) -> T:
    if isinstance(config, config_type):
        return config
    if isinstance(config, dict):
        return config_type(**config)
    raise ValueError(f"Invalid config type: {type(config)} must be {config_type} or dict that can be converted to {config_type}")

def standardize_configs(
        configs:dict[str, T|dict], 
        config_type: Type[T],
        ) -> dict[str, T]:
    std_configs = {}
    for name, config in configs.items():
        std_configs[name] = standardize_config(config, config_type)
    return std_configs    
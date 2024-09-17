from typing import Any, Sequence, Union, Optional, Literal
from unifai.types import (
    Message,
    MessageInput,
    Tool,
    ToolInput,
    ToolChoiceInput,
    ResponseFormatInput,
    EvaluateParameters,
    EvaluateParametersInput
)
from .tool_from_dict import tool_from_dict
from .tool_from_func import tool_from_func

def standardize_messages(messages: Sequence[MessageInput]) -> list[Message]:
    std_messages = []
    for message in messages:
        if isinstance(message, Message):
            std_messages.append(message)
        elif isinstance(message, str):
            std_messages.append(Message(role="user", content=message))
        elif isinstance(message, dict):
            std_messages.append(Message(**message))
        else:
            raise ValueError(f"Invalid message type: {type(message)}")        
    return std_messages


def standardize_tools(tools: Sequence[ToolInput], tool_dict: Optional[dict[str, Tool]] = None) -> dict[str, Tool]:
    std_tools = {}
    for tool in tools:            
        if isinstance(tool, dict):
            tool = tool_from_dict(tool)                        
        elif isinstance(tool, str):
            if tool_dict and (std_tool := tool_dict.get(tool)):
                tool = std_tool
            else:
                raise ValueError(f"Tool '{tool}' not found in tools")
        elif (is_not_tool := not isinstance(tool, Tool)) and callable(tool):
            tool = tool_from_func(tool)
        elif is_not_tool:
            raise ValueError(f"Invalid tool type: {type(tool)}")       
        std_tools[tool.name] = tool    
    return std_tools


def standardize_tool_choice(tool_choice: ToolChoiceInput) -> str|list[str]:
    if isinstance(tool_choice, str):
        return tool_choice if tool_choice != 'any' else 'required'
    if isinstance(tool_choice, Tool):
        return tool_choice.name
    if isinstance(tool_choice, dict):
        tool_type = tool_choice['type']
        return tool_choice[tool_type]['name']
    if isinstance(tool_choice, Sequence):
        tool_choice_str_sequence = []
        for tool_choice_item in tool_choice:
            if not isinstance(tool_choice_item, str) and isinstance(tool_choice_item, Sequence):
                raise ValueError(f"Invalid tool_choice_item type: {type(tool_choice_item)}. Nested sequences are NOT supported.")            
            tool_choice_str_sequence.append(standardize_tool_choice(tool_choice_item))
        
        return tool_choice_str_sequence
        # if any(not isinstance(tool_choice_item, str) and isinstance(tool_choice_item, Sequence) for tool_choice_item in tool_choice):
        #     raise ValueError(f"Invalid tool_choice_item type. Nested sequences are NOT supported.")
            
        # return list(map(standardize_tool_choice, tool_choice))
    
    raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}")


def standardize_response_format(response_format: ResponseFormatInput) -> str:
    if isinstance(response_format, str):
        return response_format
    if isinstance(response_format, dict):
        return response_format['type']
    raise ValueError(f"Invalid response_format type: {type(response_format)}")


def standardize_eval_prameters(eval_types: Sequence[EvaluateParametersInput]) -> dict[str, EvaluateParameters]:
    std_eval_types = {}
    for eval_type in eval_types:
        if isinstance(eval_type, EvaluateParameters):
            std_eval_types[eval_type.eval_type] = eval_type
        elif isinstance(eval_type, dict):
            std_eval_types[eval_type['eval_type']] = EvaluateParameters(**eval_type)
        else:
            raise ValueError(f"Invalid eval_type type: {type(eval_type)}")
    return std_eval_types
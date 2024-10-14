from typing import Any, Sequence, Union, Optional, Literal, Type, TypeVar
from unifai.types import (
    Message,
    MessageInput,
    Tool,
    ToolInput,
    ToolChoiceInput,
    ResponseFormatInput,
)
from .tool_from_dict import tool_from_dict
from .tool_from_func import tool_from_func

T = TypeVar("T")

def standardize_message(message: MessageInput) -> Message:
    if isinstance(message, Message):
        return message
    if isinstance(message, str):
        return Message(role="user", content=message)
    if isinstance(message, dict):
        return Message(**message)
    raise ValueError(f"Invalid message type: {type(message)}")

def standardize_messages(messages: Sequence[MessageInput]) -> list[Message]:
    return [standardize_message(message) for message in messages]

def standardize_tool(tool: ToolInput, tool_dict: Optional[dict[str, Tool]] = None) -> Tool:
    if isinstance(tool, dict):
        return tool_from_dict(tool)                        
    elif isinstance(tool, str):
        if tool_dict and (std_tool := tool_dict.get(tool)):
            return std_tool
        else:
            raise ValueError(f"Tool '{tool}' not found in tools")
    elif (is_not_tool := not isinstance(tool, Tool)) and callable(tool):
        return tool_from_func(tool)
    elif is_not_tool:
        raise ValueError(f"Invalid tool type: {type(tool)}") 
    
def standardize_tools(tools: Sequence[ToolInput], tool_dict: Optional[dict[str, Tool]] = None) -> dict[str, Tool]:
    return {tool.name: tool for tool in (standardize_tool(tool, tool_dict) for tool in tools)}
    # std_tools = {}
    # for tool in tools:            
    #     if isinstance(tool, dict):
    #         tool = tool_from_dict(tool)                        
    #     elif isinstance(tool, str):
    #         if tool_dict and (std_tool := tool_dict.get(tool)):
    #             tool = std_tool
    #         else:
    #             raise ValueError(f"Tool '{tool}' not found in tools")
    #     elif (is_not_tool := not isinstance(tool, Tool)) and callable(tool):
    #         tool = tool_from_func(tool)
    #     elif is_not_tool:
    #         raise ValueError(f"Invalid tool type: {type(tool)}")       
    #     std_tools[tool.name] = tool    
    # return std_tools


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

    raise ValueError(f"Invalid tool_choice type: {type(tool_choice)}")


def standardize_response_format(response_format: ResponseFormatInput) -> str:
    if isinstance(response_format, str):
        return response_format
    if isinstance(response_format, dict):
        return response_format['type']
    raise ValueError(f"Invalid response_format type: {type(response_format)}")


def standardize_specs(
        specs: list[type|dict], 
        spec_type: Type[T],
        key_attr: str = 'name'
        ) -> dict[str, T]:
    std_specs = {}
    for spec in specs:
        if isinstance(spec, spec_type):
            std_specs[getattr(spec, key_attr)] = spec
        elif isinstance(spec, dict):
            std_specs[spec[key_attr]] = spec_type(**spec)
        else:
            raise ValueError(f"Invalid spec type: {type(spec)} must be {spec_type} or dict that can be converted to {spec_type}")
    return std_specs

# def standardize_rag_specs(rag_specs: list[RAGSpec|dict]):
#     std_rag_specs = {}
#     for rag_spec in rag_specs:
#         if isinstance(rag_spec, RAGSpec):
#             std_rag_specs[rag_spec.spec_name] = rag_spec
#         elif isinstance(rag_spec, dict):
#             std_rag_specs[rag_spec['spec_name']] = RAGSpec(**rag_spec)
#         else:
#             raise ValueError(f"Invalid rag_spec type: {type(rag_spec)}")
#     return std_rag_specs


# def standardize_eval_specs(eval_specs: list[EvalSpec|dict]):
#     std_eval_specs = {}
#     for eval_spec in eval_specs:
#         if isinstance(eval_spec, EvalSpec):
#             std_eval_specs[eval_spec.spec_name] = eval_spec
#         elif isinstance(eval_spec, dict):
#             std_eval_specs[eval_spec['spec_name']] = EvalSpec(**eval_spec)
#         else:
#             raise ValueError(f"Invalid eval_spec type: {type(eval_spec)}")
#     return std_eval_specs




# def standardize_eval_prameters(eval_types: Sequence[EvalSpecInput]) -> dict[str, EvalSpec]:
#     std_eval_types = {}
#     for eval_type in eval_types:
#         if isinstance(eval_type, EvalSpec):
#             std_eval_types[eval_type.eval_type] = eval_type
#         elif isinstance(eval_type, dict):
#             std_eval_types[eval_type['eval_type']] = EvalSpec(**eval_type)
#         else:
#             raise ValueError(f"Invalid eval_type type: {type(eval_type)}")
#     return std_eval_types
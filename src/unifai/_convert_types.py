from json import dumps as json_dumps
from typing import Optional, Union, Sequence, Any, Literal, Mapping, TypeVar, List, Type, Tuple, Dict, Any, Callable
from unifai.types import (
    Message,
    Tool,
    FunctionTool,
    CodeInterpreterTool,
    FileSearchTool,
    ToolCall,
    ToolParameter, 
    ToolParameterType, 
    ToolValPyTypes, 
    StringToolParameter, 
    NumberToolParameter, 
    IntegerToolParameter,
    BooleanToolParameter,
    NullToolParameter,
    ArrayToolParameter,
    ObjectToolParameter,
    AnyOfToolParameter,
    EvaluateParameters,
    EvaluateParametersInput,
    ToolInput,
)

from ast import literal_eval as ast_literal_eval
from functools import wraps

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


def tool_parameter_from_dict(
        param_dict: dict, 
        param_name: Optional[str]= None,
        param_required: Optional[bool]= True
        ) -> ToolParameter:
    
    # if isinstance(required := param_dict.get('required'), bool):
    #     param_required = required

    if (anyof_param_dicts := param_dict.get('anyOf')) is not None:
        anyOf = [
            tool_parameter_from_dict(param_dict=anyof_param_dict, param_name=param_name, param_required=param_required)
            for anyof_param_dict in anyof_param_dicts
        ]
        return AnyOfToolParameter(name=param_name, required=param_required, anyOf=anyOf)

    param_type = param_dict['type']
    param_name = param_dict.get('name', param_name)
    param_description = param_dict.get('description')
    param_enum = param_dict.get('enum')
    # param_required = param_dict.get('required', param_required)

    if param_type == 'string':
        return StringToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'number':
        return NumberToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'integer':
        return IntegerToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'boolean':
        return BooleanToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    if param_type == 'null':
        return NullToolParameter(name=param_name, description=param_description, required=param_required, enum=param_enum)
    
    if param_type == 'array':
        if not (param_items := param_dict.get('items')):
            raise ValueError("Array parameters must have an 'items' key.")
        
        items = tool_parameter_from_dict(param_dict=param_items, param_required=param_required)
        return ArrayToolParameter(name=param_name, description=param_description, 
                                  required=param_required, enum=param_enum, 
                                  items=items)    
    if param_type == 'object':
        if not (param_properties := param_dict.get('properties')):
            raise ValueError("Object parameters must have a 'properties' key.")
        if isinstance(param_properties, dict):
            required_params = param_dict.get('required', [])
            properties = [
                tool_parameter_from_dict(param_dict=prop_dict, param_name=prop_name, param_required=prop_name in required_params) 
                for prop_name, prop_dict in param_dict['properties'].items()
            ]
        else:
            properties = [
                tool_parameter_from_dict(param_dict=prop_dict, param_required=prop_dict.get('required', True)) 
                for prop_dict in param_properties
            ]
        additionalProperties = param_dict.get('additionalProperties', False)
        return ObjectToolParameter(name=param_name, description=param_description, 
                                   required=param_required, enum=param_enum, 
                                   properties=properties, additionalProperties=additionalProperties)
    
    raise ValueError(f"Invalid parameter type: {param_type}")



def tool_from_dict(tool_dict: dict) -> Tool:
    tool_type = tool_dict['type']
    if tool_type == 'code_interpreter':
        return CodeInterpreterTool()
    if tool_type == 'file_search':
        return FileSearchTool()
    
    tool_def = tool_dict.get(tool_type) or tool_dict.get("input_schema")
    if tool_def is None:
        raise ValueError("Invalid tool definition. "
                         f"The input schema must be defined under the key '{tool_type}' or 'input_schema' when tool type='{tool_type}'.")

    parameters = tool_parameter_from_dict(param_dict=tool_def['parameters'], 
            # param_name='parameters',
            # param_required=True
    )
    if parameters.type == 'anyOf':
        raise ValueError("Root parameter cannot be anyOf: See: https://platform.openai.com/docs/guides/structured-outputs/root-objects-must-not-be-anyof")

    return FunctionTool(
        name=tool_def['name'], 
        description=tool_def['description'], 
        parameters=parameters,
        strict=tool_def.get('strict', True)
    )


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


def standardize_messages(messages: Sequence[Union[Message, str, dict[str, Any]]]) -> list[Message]:
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


# def standardize_tools(tools: Sequence[ToolInput], tool_dict: Optional[dict[str, Tool]] = None) -> list[Tool]:
#     std_tools = []
#     for tool in tools:
#         if isinstance(tool, Tool):
#             std_tools.append(tool)
#         elif isinstance(tool, dict):
#             std_tools.append(tool_from_dict(tool))
#         elif isinstance(tool, str):
#             if tool_dict and (std_tool := tool_dict.get(tool)):
#                 std_tools.append(std_tool)
#             else:
#                 raise ValueError(f"Tool '{tool}' not found in tools")
#         else:
#             raise ValueError(f"Invalid tool type: {type(tool)}")
    
#     return std_tools

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
        elif not isinstance(tool, Tool):
            raise ValueError(f"Invalid tool type: {type(tool)}")        
        std_tools[tool.name] = tool    
    return std_tools

def standardize_tool_choice(tool_choice: Union[Literal["auto", "required", "none"], Tool, str, dict]) -> str:
    if isinstance(tool_choice, Tool):
        return tool_choice.name
    if isinstance(tool_choice, dict):
        tool_type = tool_choice['type']
        return tool_choice[tool_type]['name']
    
    # tool_choice is a string tool_name or Literal value "auto", "required", or "none"
    return tool_choice

PY2AI_TYPES: dict[str|Type, ToolParameterType] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "None": "null",
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

def parse_docstring(docstring: str, annotations: Optional[dict]=None) -> tuple[str, ObjectToolParameter]:
    from re import compile as re_compile
    param_pattern = re_compile(r'(?P<indent>\s*)(?P<name>\w+)(?: ?\(?(?P<type>\w+)\)?)?: ?(?P<description>.+)?')

    if "Returns:" in docstring:
        docstring, returns = docstring.rsplit("Returns:", 1)
        returns = returns.strip()
    else:
        returns = ""

    if "Args:" in docstring:
        docstring, args = docstring.rsplit("Args:", 1)        
    elif "Parameters:" in docstring:
        docstring, args = docstring.rsplit("Parameters:", 1)        
    else:
        return docstring.strip(), ObjectToolParameter(properties=[])
    
    description = docstring.strip()
    args = args.rstrip()
    
    param_lines = []
    for line in args.split("\n"):
        if not (lstripped_line := line.lstrip()):
            continue
    
        if param_match := param_pattern.match(line):
            group_dict = param_match.groupdict()
            group_dict["indent"] = len(group_dict["indent"])

            # Docstring type annotations override inferred types
            if type_str := group_dict.get("type"):
                group_dict["type"] = PY2AI_TYPES.get(type_str, type_str)
            elif annotations and (anno := annotations.get(group_dict["name"])):
                group_dict["type"] = PY2AI_TYPES.get(anno)
            # else:
            #     group_dict["type"] = "string"
            param_lines.append(group_dict)
        else:
            param_lines[-1]["description"] += lstripped_line


    root = {"type": "object", "properties": []}
    stack = [root]
    for param in param_lines:                    
        # Determine the depth (number of spaces) based on the "indent" field
        param_indent = param["indent"]
        param["properties"] = [] # Initialize properties list to store nested parameters

        # If the current parameter is at the same or lower level than the last, backtrack
        while len(stack) > 1 and param_indent <= stack[-1]["indent"]:
            stack.pop()
        
        current_structure = stack[-1]
        if (param_name := param["name"]) in ("enum", "required"):
            # If the parameter is an enum or required field, add it to the current structure
            current_structure[param_name] = ast_literal_eval(param["description"])

        elif (current_type := current_structure.get("type")) == "array" and param_name == "items":
            current_structure["items"] = param
            param.pop("name") # TODO remove this line

        elif current_type == "object" and param_name == "properties":
            current_structure["properties"] = param["properties"]

        elif current_type == "anyOf" and param_name == "anyOf":
            current_structure["anyOf"] = param["properties"]            
        
        else:
            current_structure["properties"].append(param)

        stack.append(param)

    parameters = tool_parameter_from_dict(root)
    return description, parameters

    

def tool_from_function(func: Callable) -> FunctionTool:
    tool_name = func.__name__
    tool_description, tool_parameters = parse_docstring(
        docstring=func.__doc__ or "",
        annotations=func.__annotations__
        )

    return FunctionTool(
        name=tool_name,
        description=tool_description,
        parameters=tool_parameters,
        callable=func
    )

def tool(func: Callable) -> FunctionTool:
    return tool_from_function(func)



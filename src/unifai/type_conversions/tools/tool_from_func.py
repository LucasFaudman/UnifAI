from typing import Callable, Optional, Type, Collection
from ast import literal_eval as ast_literal_eval
from re import compile as re_compile

from unifai.types import Tool, ObjectToolParameter, ToolParameterType
from .construct_tool_parameter import resolve_annotation, construct_tool_parameter

PY_TYPE_TO_TOOL_PARAMETER_TYPE_MAP: dict[str|Type, ToolParameterType] = {
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

TOOL_PARAMETER_REGEX = re_compile(r'(?P<indent>\s*)(?P<name>\w+)(?: *\(?(?P<type>\w+)\)?)?: *(?P<description>.+)?')

def parse_docstring_and_annotations(
        docstring: str, 
        annotations: Optional[dict]=None,
        exclude: Optional[Collection[str]] = None,
        ) -> tuple[str, ObjectToolParameter]:
    

    if "Returns:" in docstring:
        docstring, returns = docstring.rsplit("Returns:", 1)
        returns = returns.strip()
        # TODO maybe add returns to description
    else:
        returns = ""

    if "Args:" in docstring:
        docstring, args = docstring.rsplit("Args:", 1)        
    elif "Parameters:" in docstring:
        docstring, args = docstring.rsplit("Parameters:", 1)       
    else:
        docstring, args = docstring, ""
    
    description = docstring.strip()
    args = args.rstrip()
    
    param_lines = []
    for line in args.split("\n"):
        if not (lstripped_line := line.lstrip()):
            continue
    
        if param_match := TOOL_PARAMETER_REGEX.match(line):
            param = param_match.groupdict()
            # Convert the indent to an integer
            param["indent"] = len(param["indent"])

            if type_str := param.get("type"):
                param["type"] = PY_TYPE_TO_TOOL_PARAMETER_TYPE_MAP.get(type_str, type_str)
            elif annotations and (anno := annotations.get(param["name"])):
                param["type"] = PY_TYPE_TO_TOOL_PARAMETER_TYPE_MAP.get(anno, anno)
            # else:
            #     param["type"] = "string"
            param_lines.append(param)
        else:
            # If its not a parameter line, its part of the description of the last parameter
            param_lines[-1]["description"] += lstripped_line


    root = {"type": "object", "properties": {}}
    stack = [root]
    for param in param_lines:                    
        # Determine the depth (number of spaces) based on the "indent" field
        param_indent = param["indent"]
        param["properties"] = {} # Initialize properties dict to store nested parameters

        # If the current parameter is at the same or lower level than the last, backtrack
        while len(stack) > 1 and param_indent <= stack[-1]["indent"]:
            stack.pop()
        
        current_structure = stack[-1]
        if (param_name := param["name"]) == "enum":
            # If the parameter is an enum, add it to the current structure
            current_structure[param_name] = ast_literal_eval(param["description"])
        elif (current_type := current_structure.get("type")) == "array" and param_name == "items":
            current_structure["items"] = param
            param.pop("name") # TODO remove this line
        elif current_type == "object" and param_name == "properties":
            current_structure["properties"] = param["properties"]
        elif current_type == "anyOf" and param_name == "anyOf":
            current_structure["anyOf"] = param["properties"]            
        else:
            current_structure["properties"][param_name] = param

        stack.append(param)

    # Annotations override docstring
    if annotations:
        for param_name, anno in annotations.items():
            if param_name == "return" or (exclude and param_name in exclude):
                continue
            if param_name not in root["properties"]:
                root["properties"][param_name] = resolve_annotation(anno)

    parameters = construct_tool_parameter(root)
    assert isinstance(parameters, ObjectToolParameter)
    return description, parameters


def tool_from_func(
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: str = "function",
        strict: bool = True,
        exclude: Optional[Collection[str]] = None,
    ) -> Tool:
    docstring_description, parameters = parse_docstring_and_annotations(
        docstring=func.__doc__ or "",
        annotations=func.__annotations__,
        exclude=exclude
        )
    return Tool(
        name=name or func.__name__,
        description=description or docstring_description,
        parameters=parameters,
        type=type,
        strict=strict,
        callable=func
    )
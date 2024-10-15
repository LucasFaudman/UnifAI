from typing import Callable, Optional, Type
from ast import literal_eval as ast_literal_eval

from unifai.types import Tool, ObjectToolParameter, ToolParameterType
from .tool_from_dict import tool_parameter_from_dict

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

def parse_docstring_and_annotations(
        docstring: str, 
        annotations: Optional[dict]=None
        ) -> tuple[str, ObjectToolParameter]:
    
    from re import compile as re_compile
    param_pattern = re_compile(r'(?P<indent>\s*)(?P<name>\w+)(?: *\(?(?P<type>\w+)\)?)?: *(?P<description>.+)?')

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
                group_dict["type"] = PY_TYPE_TO_TOOL_PARAMETER_TYPE_MAP.get(type_str, type_str)
            elif annotations and (anno := annotations.get(group_dict["name"])):
                group_dict["type"] = PY_TYPE_TO_TOOL_PARAMETER_TYPE_MAP.get(anno)
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
    assert isinstance(parameters, ObjectToolParameter)
    return description, parameters


def tool_from_func(
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: str = "function",
        strict: bool = True,
    ) -> Tool:
    docstring_description, parameters = parse_docstring_and_annotations(
        docstring=func.__doc__ or "",
        annotations=func.__annotations__
        )
    return Tool(
        name=name or func.__name__,
        description=description or docstring_description,
        parameters=parameters,
        type=type,
        strict=strict,
        callable=func
    )


# def tool_from_func(func: Callable) -> Tool:
#     name = func.__name__
#     description, parameters = parse_docstring_and_annotations(
#         docstring=func.__doc__ or "",
#         annotations=func.__annotations__
#         )

#     return Tool(
#         name=name,
#         description=description,
#         parameters=parameters,
#         callable=func
#     )

# # alias for tool_from_func so functions can be decorated with @tool or @tool_from_func
# tool = tool_from_func
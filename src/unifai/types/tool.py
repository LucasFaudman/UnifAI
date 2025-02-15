from typing import Optional, Union, Sequence, Any, Literal, Callable, Generic, Collection
from ._base_model import BaseModel

from ..utils import clean_locals
from .annotations import InputP, ReturnT
from .tool_parameters import ToolParameter, ObjectToolParameter, ToolParameterExcludableKeys, EXCLUDE_NONE

ToolExcludableKeys = ToolParameterExcludableKeys | Literal["strict"]

class Tool(BaseModel, Generic[InputP, ReturnT]):
    type: str = "function"
    name: str
    description: str
    parameters: ObjectToolParameter
    strict: bool = True
    callable: Optional[Callable[InputP, ReturnT]] = None

    def __init__(self, 
        name: str, 
        description: str, 
        *args: ToolParameter,
        parameters: Optional[ObjectToolParameter | dict[str, ToolParameter] | Sequence[ToolParameter]] = None,
        type: str = "function",
        strict: bool = True,
        callable: Optional[Callable[InputP, ReturnT]] = None
    ):        
        if bool(args) ^ bool(parameters):
            raise ValueError(f"Must provide either parameters or args, not both or neither. Got: {args=}, {parameters=}")
        if args:
            parameters = ObjectToolParameter(properties=args)
        elif not isinstance(parameters, (ObjectToolParameter)):
            parameters = ObjectToolParameter(properties=parameters or ()) 
        BaseModel.__init__(self, **clean_locals(locals()))
        

    def __call__(self, *args: InputP.args, **kwargs: InputP.kwargs) -> ReturnT:
        if self.callable is None:
            raise ValueError(f"Callable not set for tool {self.name}")
        return self.callable(*args, **kwargs)

    def to_dict(
            self, 
            exclude: Collection[ToolExcludableKeys] = EXCLUDE_NONE,
            type: Optional[str] = None,
            parameters_key: str = "parameters"
        ) -> dict[str, Any]:
        _def: dict[str, Any] = { "name": self.name }
        if "description" not in exclude:
            _def["description"] = self.description
        if "strict" not in exclude:
            _def["strict"] = self.strict
        if parameters_key not in exclude:
            _def[parameters_key] = self.parameters.to_dict(exclude)
        _type = type or self.type
        return { "type": _type, _type: _def }
    
    def to_json_schema(self, exclude: Collection[ToolExcludableKeys] = EXCLUDE_NONE) -> dict[str, Any]:
        return self.to_dict(exclude, type="json_schema", parameters_key="json_schema")


class ProviderTool(Tool):
    def to_dict(self, exclude: Collection[ToolExcludableKeys] = EXCLUDE_NONE) -> dict:
        return { "type": self.type }


PROVIDER_TOOLS = {
    "code_interpreter": ProviderTool(
        type="code_interpreter", 
        name="code_interpreter", 
        description="A Python Code Interpreter Tool Implemented by OpenAI", 
        parameters=ObjectToolParameter(properties=())
    ),
    "file_search": ProviderTool(
        type="file_search", 
        name="file_search", 
        description="A File Search Tool Implemented by OpenAI", 
        parameters=ObjectToolParameter(properties=())
    ),
}
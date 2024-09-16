from typing import Optional, Union, Sequence, Any, Literal, Callable, Mapping
from pydantic import BaseModel

from .tool_parameter import ToolParameter, ObjectToolParameter, ArrayToolParameter


class Tool(BaseModel):
    type: str = "function"
    name: str
    description: str
    parameters: ObjectToolParameter
    strict: bool = True
    callable: Optional[Callable] = None

    def __init__(self, 
        name: str, 
        description: str, 
        *args: ToolParameter,
        parameters: Optional[Union[ObjectToolParameter, Sequence[ToolParameter], Mapping[str, ToolParameter]]] = None,
        type: str = "function",
        strict: bool = True,
        callable: Optional[Callable] = None
    ):        
             
        if isinstance(parameters, ObjectToolParameter):
            parameters = parameters
            # parameters.name = "parameters"
        elif isinstance(parameters, Sequence):
            parameters = ObjectToolParameter(properties=parameters)
        elif isinstance(parameters, Mapping):
            properties = []
            for parameter_name, parameter in parameters.items():
                if parameter.name is None:
                    parameter.name = parameter_name
                elif parameter.name != parameter_name:
                    raise ValueError("Parameter name does not match key")
                properties.append(parameter)
            parameters = ObjectToolParameter(properties=properties)
        elif args:
            parameters = ObjectToolParameter(properties=list(args))
        else:
            raise ValueError(f"Invalid parameters type: {parameters}")

        BaseModel.__init__(self, name=name, type=type, description=description, parameters=parameters, strict=strict, callable=callable)


    def __call__(self, *args, **kwargs) -> Any:
        if self.callable is None:
            raise ValueError(f"Callable not set for tool {self.name}")
        return self.callable(*args, **kwargs)


    def to_dict(self):
        return {
            "type": self.type,
            self.type: {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.to_dict(),
                "strict": self.strict
            },            
        }


class ProviderTool(Tool):
    def to_dict(self) -> dict:
        return {
            "type": self.type,
        }


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
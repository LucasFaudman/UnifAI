from typing import Generic
from .._base_components._base_function import BaseFunction, FunctionConfig, InputT, OutputT, ReturnT

class Function(BaseFunction[FunctionConfig, InputT, OutputT, ReturnT], Generic[InputT, OutputT, ReturnT]):
    component_type = "function"
    provider = "default"
    config_class = FunctionConfig
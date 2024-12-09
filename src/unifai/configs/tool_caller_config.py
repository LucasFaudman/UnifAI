from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar
from ._base_configs import ComponentConfig

class ToolCallerConfig(ComponentConfig):
    component_type: ClassVar = "tool_caller"
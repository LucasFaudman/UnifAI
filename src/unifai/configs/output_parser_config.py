from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar
from ._base_configs import ComponentConfig

class OutputParserConfig(ComponentConfig):
    component_type: ClassVar = "output_parser"
    extra_kwargs: Optional[dict[Literal["parse"], dict[str, Any]]] = None
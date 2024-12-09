from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_output_parser import OutputParser
    from ..configs.output_parser_config import OutputParserConfig, OutputT, ReturnT
    from ..types.annotations import ComponentName, ProviderName

from ._base_client import BaseClient

class UnifAIOutputParserClient(BaseClient):
    
    def get_output_parser(
            self, 
            provider_config_or_name: "ProviderName | OutputParserConfig[OutputT,ReturnT] | tuple[ProviderName, ComponentName]" = "default",          
            **init_kwargs
            ) -> "OutputParser[OutputT,ReturnT]":
        return self._get_component("output_parser", provider_config_or_name, init_kwargs)
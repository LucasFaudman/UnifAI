from typing import Generic, Callable, Any, Type, cast

from ...configs.input_parser_config import InputParserConfig
from ...configs.output_parser_config import OutputParserConfig
from ...configs.function_config import FunctionConfig, return_last_message
from ...configs.rag_config import RAGConfig

from ...types.annotations import InputP, InputReturnT, OutputT, ReturnT, NewInputP, NewInputReturnT, NewOutputT, NewReturnT
from .._base_components._base_function import BaseFunction

class Function(BaseFunction[FunctionConfig[InputP, InputReturnT, OutputT, ReturnT], InputP, InputReturnT, OutputT, ReturnT], Generic[InputP, InputReturnT, OutputT, ReturnT]):
    component_type = "function"
    provider = "default"
    config_class = FunctionConfig

    def set_input_parser(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[..., NewInputP] | 
                    FunctionConfig[NewInputP, Any, Any, NewInputReturnT]):
        self._set_input_parser(input_parser)
        return cast("Function[NewInputP, NewInputReturnT, OutputT, ReturnT]", self)
    
    def set_output_parser(self, output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     FunctionConfig[..., Any, Any, NewReturnT] = return_last_message
                     ):        
        self._set_output_parser(output_parser)
        return cast("Function[InputP, InputReturnT, NewOutputT, NewReturnT]", self)
    
    def set_parsers(self, input_parser: Callable[NewInputP, NewInputReturnT | Callable[..., NewInputReturnT]] | 
                    InputParserConfig[NewInputP, NewInputReturnT] | 
                    RAGConfig[..., NewInputP] | 
                    FunctionConfig[NewInputP, Any, Any, NewInputReturnT],
                    output_parser: Type[NewReturnT] | 
                     Callable[[NewOutputT], NewReturnT] | 
                     OutputParserConfig[NewOutputT, NewReturnT] | 
                     FunctionConfig[..., Any, Any, NewReturnT] = return_last_message
        ):
        self = self.set_input_parser(input_parser)
        self = self.set_output_parser(output_parser)
        return self
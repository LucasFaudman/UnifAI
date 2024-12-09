from typing import Any
from importlib import import_module

def lazy_import(module_name: str) -> Any:
    module_name, *submodules = module_name.split(".")
    if not (module := globals().get(module_name)):
        # TODO - ClientImportError
        module = import_module(module_name)
        globals()[module_name] = module
                
    for submodule in submodules:
        module = getattr(module, submodule)        
    return module



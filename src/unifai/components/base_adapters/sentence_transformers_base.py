from typing import Any
from ._base_adapter import UnifAIAdapter, UnifAIComponent
from importlib import import_module

class SentenceTransformersAdapter(UnifAIAdapter):
    provider = "sentence_transformers"
   
    def lazy_import(self, module_name: str) -> Any:
        module_name, *submodules = module_name.split(".")
        if not (module := globals().get(module_name)):
            module = import_module(module_name)
            globals()[module_name] = module
                    
        for submodule in submodules:
            module = getattr(module, submodule)        
        return module


    def import_client(self):
        return self.lazy_import("sentence_transformers")


    def init_client(self, **client_kwargs):
        self.client_kwargs.update(client_kwargs)
    

    # List Models
    def list_models(self) -> list[str]:
        hugging_face_api = self.lazy_import('huggingface_hub.HfApi')()
        return hugging_face_api.list_models(library="sentence-transformers")
  
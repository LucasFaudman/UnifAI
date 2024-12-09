from typing import Any
from .._base_components._base_adapter import UnifAIAdapter, UnifAIComponent
from ...utils import lazy_import

class SentenceTransformersAdapter(UnifAIAdapter):
    provider = "sentence_transformers"
   
    def import_client(self):
        return lazy_import("sentence_transformers")

    def init_client(self, **client_kwargs):
        self.client_kwargs.update(client_kwargs)
    
    # List Models
    def list_models(self) -> list[str]:
        hugging_face_api = lazy_import('huggingface_hub.HfApi')()
        return hugging_face_api.list_models(library="sentence-transformers")
  
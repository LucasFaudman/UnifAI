from .._base_components._base_ragpipe import BaseRAGPipe
from ...configs.rag_config import RAGConfig, InputP
from typing import Generic

class RAGPipe(BaseRAGPipe[RAGConfig, InputP], Generic[InputP]):
    component_type = "ragpipe"
    provider = "default"
    config_class = RAGConfig
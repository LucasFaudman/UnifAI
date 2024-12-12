from .._base_components._base_ragpipe import BaseRAGPipe, RAGConfig

class RAGPipe(BaseRAGPipe[RAGConfig]):
    component_type = "ragpipe"
    provider = "default"
    config_class = RAGConfig
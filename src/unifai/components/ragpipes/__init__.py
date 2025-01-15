from .._base_components._base_ragpipe import BaseRAGPipe
from .._base_components._base_rag_prompter import BaseRAGPrompter
from ...configs.rag_config import RAGConfig, InputP, RAGPrompterConfig
from typing import Generic

class RAGPipe(BaseRAGPipe[RAGConfig, InputP], Generic[InputP]):
    component_type = "ragpipe"
    provider = "default"
    config_class = RAGConfig


class RAGPrompter(BaseRAGPrompter[RAGPrompterConfig, InputP], Generic[InputP]):
    component_type = "rag_prompter"
    provider = "default"
    config_class = RAGPrompterConfig
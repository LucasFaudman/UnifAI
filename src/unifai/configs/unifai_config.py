from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self

from ..types.annotations import ComponentType, ComponentName, ModelName, ProviderName, ToolName, ToolInput, BaseModel

from ._base_configs import ProviderConfig, ComponentConfig
from .chat_config import ChatConfig
from .llm_config import LLMConfig
from .function_config import FunctionConfig
from .rag_config import RAGConfig
from .tool_caller_config import ToolCallerConfig

class UnifAIConfig(BaseModel):
    api_keys: Optional[dict[ProviderName, str]] = None
    provider_configs: Optional[list[ProviderConfig]] = None
    default_providers: Optional[dict[ComponentType, ProviderName]] =None
    component_configs: Optional[list[ComponentConfig]] = None
    chat_configs: Optional[list[ChatConfig]] = None
    function_configs: Optional[list[FunctionConfig]] = None
    rag_configs: Optional[list[RAGConfig]] = None
    tools: Optional[list[ToolInput]] = None
    tool_callables: Optional[dict[ToolName, Callable]] = None
    # document_chunkers: Optional[list[DocumentChunkerConfig]] = None
    # document_loaders: Optional[list[DocumentLoaderConfig]] = None
    # document_dbs: Optional[list[DocumentDBConfig]] = None
    # embedders: Optional[list[EmbedderConfig]] = None
    # llms: Optional[list[LLMConfig]] = None
    # output_parsers: Optional[list[OutputParserConfig]] = None
    # rerankers: Optional[list[RerankerConfig]] = None
    # tokenizers: Optional[list[TokenizerConfig]] = None
    # tool_callers: Optional[list[ToolCallerConfig]] = None
    # vector_dbs: Optional[list[VectorDBConfig]] = None

UnifAIConfig(
    provider_configs=[
        ProviderConfig(
            provider="openai", 
            api_key="123", 
            client_init_kwargs={"base_url": "https://api.openai.com/v1"}
        ),
        ProviderConfig(
            provider="google", 
            # api_key="456", 
            # client_init_kwargs={"base_url": "https://api.google.com/v1"}
        ),
    ]
)
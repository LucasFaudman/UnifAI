from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

from ..components.llms._base_llm_client import LLMClient
from ..components.embedders._base_embedder import Embedder
from ..components.rerankers._base_reranker import Reranker
from ..components.document_dbs._base_document_db import DocumentDB
from ..components.retrievers._base_vector_db_client import VectorDBClient, VectorDBIndex
from ..components.import_component import import_component, LLMS, EMBEDDERS, VECTOR_DBS, RERANKERS, DOCUMENT_DBS, PROVIDERS
from ..components.tool_callers import ToolCaller

from pathlib import Path
from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    api_key: Optional[str] = None
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    default_llm_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_rerank_model: Optional[str] = None
    component_type_client_kwarg_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

class Config(BaseModel):
    provider_configs: dict[str, ProviderConfig] = Field(default_factory=dict)
    default_llm_provider: Optional[str] = None
    default_embedding_provider: Optional[str] = None
    default_vector_db_provider: Optional[str] = None
    default_rerank_provider: Optional[str] = None
    default_document_db_provider: Optional[str] = None
    default_document_chunker_provider: Optional[str] = None   


CONFIG: Config | None = None

DEFAULT_PROVIDERS = {
    "llm": "openai",
    "embedding": "openai",
    "vector_db": "chroma",
    "rerank": "cohere",
    "document_db": "dict",
    "document_chunker": "unstructured"
}

COMPONENT_REGISTRY: dict[str, dict[str, Type[LLMClient|Embedder|VectorDBClient|Reranker|DocumentDB]]] = {
    "llm": {},
    "embedder": {},
    "vector_db": {},
    "reranker": {},
    "document_db": {},
    "document_chunker": {}
}

COMPONENTS: dict[str, dict[str, LLMClient|Embedder|VectorDBClient|Reranker|DocumentDB]] = {
    "llm": {},
    "embedder": {},
    "vector_db": {},
    "reranker": {},
    "document_db": {},
    "document_chunker": {}
}

def get_config() -> Config:
    global CONFIG
    if CONFIG is None:
        CONFIG = Config()
    return CONFIG

def set_config(config: Config) -> None:
    global CONFIG
    CONFIG = config

def configure(
        config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
        provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
        api_keys: Optional[dict[str, str]] = None,
        default_llm_provider: Optional[str] = None,
        default_embedding_provider: Optional[str] = None,
        default_vector_db_provider: Optional[str] = None,
        default_rerank_provider: Optional[str] = None,
        default_document_db_provider: Optional[str] = None,
        default_document_chunker_provider: Optional[str] = None,           
        **kwargs        
    ):
    
    if config_obj_dict_or_path is not None:
        if isinstance(config_obj_dict_or_path, str):
            config_obj_dict_or_path = Path(config_obj_dict_or_path)
        if isinstance(config_obj_dict_or_path, Path):
            config = Config.model_validate_json(config_obj_dict_or_path.read_bytes())
        elif isinstance(config_obj_dict_or_path, dict):
            config = Config.model_validate(config_obj_dict_or_path)
        elif isinstance(config_obj_dict_or_path, Config):
            config = config_obj_dict_or_path
        else:
            raise ValueError(f"Invalid config type: {type(config_obj_dict_or_path)}. Must be one of: unifai.Config, dict, str, Path")        
        set_config(config)
    
    config = get_config()

    if provider_configs is not None:
        for provider, provider_config in provider_configs.items():
            if isinstance(provider_config, dict):
                provider_config = ProviderConfig.model_validate(provider_config)
            config.provider_configs[provider] = provider_config

    if api_keys is not None:
        for provider, api_key in api_keys.items():
            if provider_config := config.provider_configs.get(provider):
                provider_config.api_key = api_key
            else:
                config.provider_configs[provider] = ProviderConfig(api_key=api_key)

    if default_llm_provider is not None:
        config.default_llm_provider = default_llm_provider
    if default_embedding_provider is not None:
        config.default_embedding_provider = default_embedding_provider
    if default_vector_db_provider is not None:
        config.default_vector_db_provider = default_vector_db_provider
    if default_rerank_provider is not None:
        config.default_rerank_provider = default_rerank_provider
    if default_document_db_provider is not None:
        config.default_document_db_provider = default_document_db_provider
    if default_document_chunker_provider is not None:
        config.default_document_chunker_provider = default_document_chunker_provider
    

def register_component(
        provider: str,
        component_type: Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker"],
        component_class: Type[LLMClient|Embedder|VectorDBClient|Reranker|DocumentDB]
    ) -> None:
    if component_type not in COMPONENT_REGISTRY:
        COMPONENT_REGISTRY[component_type] = {}
    COMPONENT_REGISTRY[component_type][provider] = component_class


def init_component(
        provider: str,
        component_type: Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker"],
        **client_kwargs
    ) -> LLMClient|Embedder|VectorDBClient|Reranker|DocumentDB:
    if component_type not in COMPONENT_REGISTRY:
        raise ValueError(f"Invalid component_type: {component_type}. Must be one of: {', '.join(COMPONENT_REGISTRY.keys())}")
    if (component_class := COMPONENT_REGISTRY[component_type].get(provider)) is None:
        component_class = import_component(provider, component_type)
        register_component(provider, component_type, component_class)
    
    component_kwargs = {}
    if provider_config := get_config().provider_configs.get(provider):
        component_kwargs.update(provider_config.client_kwargs)
        if provider_config.api_key:
            component_kwargs["api_key"] = provider_config.api_key
        if component_type_client_kwargs := provider_config.component_type_client_kwarg_overrides.get(component_type):
            component_kwargs.update(component_type_client_kwargs)
    component_kwargs.update(client_kwargs)
    
    component = component_class(**component_kwargs)
    COMPONENTS[component_type][provider] = component
    return component


@overload
def get_component(provider: str, component_type: Literal["llm"], **client_kwargs) -> LLMClient:
    ...

@overload
def get_component(provider: str, component_type: Literal["embedder"], **client_kwargs) -> Embedder:
    ...

@overload
def get_component(provider: str, component_type: Literal["vector_db"], **client_kwargs) -> VectorDBClient:
    ...

@overload
def get_component(provider: str, component_type: Literal["reranker"], **client_kwargs) -> Reranker:
    ...

@overload
def get_component(provider: str, component_type: Literal["document_db"], **client_kwargs) -> DocumentDB:
    ...

@overload
def get_component(provider: str, component_type: Literal["document_chunker"], **client_kwargs) -> DocumentDB:
    # TODO - Implement document_chunker
    ...
    
def get_component(
        provider: str,
        component_type: Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker"],
        **client_kwargs
    ) -> LLMClient|Embedder|VectorDBClient|Reranker|DocumentDB:
    components_of_type = COMPONENTS.get(component_type)
    if not components_of_type or provider not in components_of_type:
        return init_component(provider, component_type, **client_kwargs)
    return COMPONENTS[component_type][provider]


def get_llm_client(provider: Optional[str] = None, **client_kwargs) -> LLMClient:
    provider = provider or get_config().default_llm_provider or DEFAULT_PROVIDERS["llm"]
    return get_component(provider, "llm", **client_kwargs)

def get_embedder(provider: Optional[str] = None, **client_kwargs) -> Embedder:
    provider = provider or get_config().default_embedding_provider or DEFAULT_PROVIDERS["embedding"]
    return get_component(provider, "embedder", **client_kwargs)

def get_reranker(provider: Optional[str] = None, **client_kwargs) -> Reranker:
    provider = provider or get_config().default_rerank_provider or DEFAULT_PROVIDERS["rerank"]
    return get_component(provider, "reranker", **client_kwargs)

def get_vector_db(provider: Optional[str] = None, **client_kwargs) -> VectorDBClient:
    provider = provider or get_config().default_vector_db_provider or DEFAULT_PROVIDERS["vector_db"]
    return get_component(provider, "vector_db", **client_kwargs)

def get_document_db(provider: Optional[str] = None, **client_kwargs) -> DocumentDB:
    provider = provider or get_config().default_document_db_provider or DEFAULT_PROVIDERS["document_db"]
    return get_component(provider, "document_db", **client_kwargs)

def get_document_chunker(provider: Optional[str] = None, **client_kwargs) -> DocumentDB:
    provider = provider or get_config().default_document_chunker_provider or DEFAULT_PROVIDERS["document_chunker"]
    return get_component(provider, "document_chunker", **client_kwargs)


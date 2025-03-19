from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..types.documents import Document, Documents
    from ..components._base_components._base_embedder import Embedder, Embeddings

from ..utils import copy_init_from
from ..configs.embedder_config import EmbedderConfig
from ._base_client import BaseClient

    
class UnifAIEmbedClient(BaseClient):

    def _get_embedder(
            self, 
            config_or_name: "EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",      
            **init_kwargs
            ) -> "Embedder":
        return self._get_component("embedder", config_or_name, init_kwargs)

    def embedder_from_config(
            self, 
            config: "EmbedderConfig",
            **init_kwargs
            ) -> "Embedder":
        return self._get_component("embedder", config, init_kwargs)
    
    def embedder_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ) -> "Embedder":
        return self._get_component("embedder", (provider, name), init_kwargs)

    @copy_init_from(EmbedderConfig.__init__)
    def embedder(self, *args, **kwargs) -> "Embedder":
        config = EmbedderConfig(*args, **kwargs)
        return self._get_embedder(config)

    # Embeddings
    def embed(
        self,            
        input: str | list[str],
        config_or_name: "EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",
        model: Optional[str] = None,        
        dimensions: Optional[int] = None,
        task_type: Optional[Literal[
            "retrieval_document", 
            "retrieval_query", 
            "semantic_similarity", 
            "classification", 
            "clustering", 
            "question_answering", 
            "fact_verification", 
            "code_retrieval_query", 
            "image"]] = None,
        truncate: Literal[False, "end", "start"] = False,
        reduce_dimensions: bool = False,
        use_closest_supported_task_type: bool = True,        
        **kwargs
        ) -> "Embeddings":
        
        return self._get_embedder(config_or_name).embed(
            input,
            model=model,
            dimensions=dimensions,
            task_type=task_type,
            truncate=truncate,
            reduce_dimensions=reduce_dimensions,
            use_closest_supported_task_type=use_closest_supported_task_type,
            **kwargs
        )

    def embed_documents(
            self,            
            documents: Iterable["Document"],
            config_or_name: "EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            model: Optional[str] = None,
            dimensions: Optional[int] = None,
            task_type: Optional[Literal[
                "retrieval_document", 
                "retrieval_query", 
                "semantic_similarity", 
                "classification", 
                "clustering", 
                "question_answering", 
                "fact_verification", 
                "code_retrieval_query", 
                "image"]] = None,
            truncate: Literal[False, "end", "start"] = False,
            reduce_dimensions: bool = False,
            use_closest_supported_task_type: bool = True,  
            batch_size: Optional[int] = None,           
            **kwargs
            ) -> "Documents":
        return self._get_embedder(config_or_name).embed_documents(
            documents,
            model=model,
            dimensions=dimensions,
            task_type=task_type,
            truncate=truncate,
            reduce_dimensions=reduce_dimensions,
            use_closest_supported_task_type=use_closest_supported_task_type,
            batch_size=batch_size,
            **kwargs
        )
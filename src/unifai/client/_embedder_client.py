from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..types.documents import Document, Documents
    from ..components._base_components._base_embedder import Embedder, Embeddings
    from ..configs.embedder_config import EmbedderConfig

from ._base_client import BaseClient

    
class UnifAIEmbedClient(BaseClient):

    def embedder(
            self, 
            provider_config_or_name: "ProviderName | EmbedderConfig | tuple[ProviderName, ComponentName]" = "default",      
            **init_kwargs
            ) -> "Embedder":
        return self._get_component("embedder", provider_config_or_name, init_kwargs)

    # Embeddings
    def embed(
        self,            
        input: str | list[str],
        provider_config_or_name: "ProviderName | EmbedderConfig | tuple[ProviderName, ComponentName]" = "default",
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
        
        return self.embedder(provider_config_or_name).embed(
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
            provider_config_or_name: "ProviderName | EmbedderConfig | tuple[ProviderName, ComponentName]" = "default",
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
        return self.embedder(provider_config_or_name).embed_documents(
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
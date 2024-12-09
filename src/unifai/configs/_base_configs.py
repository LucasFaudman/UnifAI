from typing import TYPE_CHECKING, Any, Literal, Optional, Type, TypeAlias, Annotated, Generic, TypeVar, ClassVar
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self

from ..types import (BaseModel, Field, ComponentName, ModelName,  ProviderName, CollectionName)


class BaseConfig(BaseModel):
    name: ComponentName = "default"

class ProviderConfig(BaseModel):    
    provider: ProviderName
    api_key: Optional[str] = None
    client_init_kwargs: dict[str, Any] = Field(default_factory=dict)
    share_client: bool = True

class ComponentConfig(BaseConfig):
    component_type: ClassVar = "component"    
    provider: ProviderName
    init_kwargs: dict[str, Any] = Field(default_factory=dict)

    def _get_other_component_type(self, other: "ComponentConfig") -> str:
        if not (other_component_type := getattr(other, "component_type", None)):
            self_name, other_name = self.__class__.__name__, other.__class__.__name__            
            raise ValueError(f"Cannot add {other_name} to {self_name}. "
                             "Only ComponentConfig objects can be added to other ComponentConfig objects.")
        if not hasattr(self, other_component_type):
            self_name, other_name = self.__class__.__name__, other.__class__.__name__            
            raise ValueError(f"Cannot add {other_name} to {self_name} since {self_name} does not a use an {other_name} object."
                             f"{other_name} objects can only be added to ComponentConfig objects that have an attribute named {other_component_type}."
                             )
        return other_component_type
    
    def __add__(self, other: "ComponentConfig") -> "ComponentConfig":
        return self.model_copy(update={self._get_other_component_type(other): other}, deep=True)

    def __iadd__(self, other: "ComponentConfig") -> Self:
        setattr(self, self._get_other_component_type(other), other)
        return self

class ComponentWithModelConfig(ComponentConfig):
    default_model: Optional[ModelName] = None
        
class BaseDBCollectionConfig(ComponentConfig):
    component_type: ClassVar = "base_db_collection"
    name: CollectionName = "default_collection"
    
class BaseDBConfig(ComponentConfig):
    component_type: ClassVar = "base_db"
    default_collection: BaseDBCollectionConfig
    
class BaseDocumentCleanerConfig(ComponentConfig):
    component_type: ClassVar = "base_document_cleaner"
    replacements: dict[str | Pattern, str] = {
            r'.\x08': '', # Remove backspace formatting
            r'[\x00-\x08\x0B-\x1F\x7F-\x9F]+': ' ', # Replace common control chars with space
        }
    strip_chars: Optional[str | Literal[False]] = None # call .strip(None) on text. If False, do not call .strip()    
    
    add_to_metadata: Optional[list[Literal["source", "chunk_size", "start_index", "end_index"]]] = ["source"]    
    deepcopy_metadata: bool = True




 





    


    











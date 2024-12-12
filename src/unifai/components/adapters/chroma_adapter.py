from typing import Any

from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings as ChromaSettings
from chromadb.db.base import UniqueConstraintError
from chromadb.errors import ChromaError, InvalidCollectionException, InvalidDimensionException
from chromadb.api import ClientAPI as ChromaClientAPI

from ...exceptions import UnifAIError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, CollectionAlreadyExistsError, CollectionNotFoundError, EmbeddingDimensionsError
from .._base_components._base_component import UnifAIComponent
from .._base_components._base_adapter import UnifAIAdapter
 

class ChromaExceptionConverter(UnifAIComponent):
    provider = "chroma"

    def _convert_exception(self, exception: ChromaError|UniqueConstraintError) -> UnifAIError:
        if isinstance(exception, InvalidCollectionException):
            return CollectionNotFoundError(
                message=exception.message(),
                original_exception=exception,
                status_code=exception.code(),
                error_code=exception.name()
            )
        if isinstance(exception, InvalidDimensionException):
            return EmbeddingDimensionsError(
                message=exception.message(),
                original_exception=exception,
                status_code=exception.code(),
                error_code=exception.name()
            )

        if isinstance(exception, UniqueConstraintError):
            message = exception.args[0]
            if "Collection" in message and "already exists" in message:
                return CollectionAlreadyExistsError(
                    message=message,
                    original_exception=exception
                )
            return UnknownAPIError(
                message=message,
                original_exception=exception
            )
        
        status_code=getattr(exception, "code", -1)
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=getattr(exception, "message", str(exception)),
            status_code=status_code,
            original_exception=exception
        )


class ChromaAdapter(UnifAIAdapter, ChromaExceptionConverter):
    client: ChromaClientAPI
    default_embedding_provider = "ollama"


    def import_client(self) -> Any:
        from chromadb import Client
        return Client

        
    def init_client(self, **init_kwargs) -> ChromaClientAPI:
        self.init_kwargs.update(init_kwargs)
        # tentant = self.init_kwargs.get("tenant", DEFAULT_TENANT)
        # database = self.init_kwargs.get("database", DEFAULT_DATABASE)
        path = self.init_kwargs.pop("path", None)
        settings = self.init_kwargs.get("settings", None)

        extra_kwargs = {k: v for k, v in self.init_kwargs.items() if k not in ["tenant", "database", "settings"]}

        if settings is None:
            settings = ChromaSettings(**extra_kwargs)
        elif isinstance(settings, dict):
            settings = ChromaSettings(**settings, **extra_kwargs)
        elif not isinstance(settings, ChromaSettings):
            raise ValueError("Settings must be a dictionary or a chromadb.config.Settings object")

        for k in extra_kwargs:
            setattr(settings, k, self.init_kwargs.pop(k))

        if path is not None:
            if settings.persist_directory:
                raise ValueError("Path and persist_directory cannot both be set. path is shorthand for persist_directory={path} and is_persistent=True")
            settings.persist_directory = path if isinstance(path, str) else str(path)
            settings.is_persistent = True
        elif settings.persist_directory and not settings.is_persistent:
            settings.is_persistent = True           

        self.init_kwargs["settings"] = settings
        self._client = self.import_client()(**self.init_kwargs)
        return self._client
   
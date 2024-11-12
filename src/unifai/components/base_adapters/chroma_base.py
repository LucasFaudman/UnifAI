from typing import Any

from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings as ChromaSettings
from chromadb.db.base import UniqueConstraintError
from chromadb.errors import ChromaError
from chromadb.api import ClientAPI as ChromaClientAPI

from ...exceptions import UnifAIError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, IndexAlreadyExistsError, IndexNotFoundError
from .._base_component import UnifAIComponent
from ._base_adapter import UnifAIAdapter
 

class ChromaExceptionConverter(UnifAIComponent):
    def convert_exception(self, exception: ChromaError|UniqueConstraintError) -> UnifAIError:
        if isinstance(exception, UniqueConstraintError):
            message = exception.args[0]
            if "Collection" in message and "already exists" in message:
                return IndexAlreadyExistsError(
                    message=message,
                    original_exception=exception
                )
            return UnknownAPIError(
                message=message,
                original_exception=exception
            )
        
        status_code=exception.code()
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=exception.message(), 
            status_code=status_code,
            original_exception=exception
        )   


class ChromaAdapter(UnifAIAdapter, ChromaExceptionConverter):
    client: ChromaClientAPI
    default_embedding_provider = "ollama"


    def import_client(self) -> Any:
        from chromadb import Client
        return Client

        
    def init_client(self, **client_kwargs) -> ChromaClientAPI:
        self.client_kwargs.update(client_kwargs)
        # tentant = self.client_kwargs.get("tenant", DEFAULT_TENANT)
        # database = self.client_kwargs.get("database", DEFAULT_DATABASE)
        path = self.client_kwargs.pop("path", None)
        settings = self.client_kwargs.get("settings", None)

        extra_kwargs = {k: v for k, v in self.client_kwargs.items() if k not in ["tenant", "database", "settings"]}

        if settings is None:
            settings = ChromaSettings(**extra_kwargs)
        elif isinstance(settings, dict):
            settings = ChromaSettings(**settings, **extra_kwargs)
        elif not isinstance(settings, ChromaSettings):
            raise ValueError("Settings must be a dictionary or a chromadb.config.Settings object")

        for k in extra_kwargs:
            setattr(settings, k, self.client_kwargs.pop(k))

        if path is not None:
            if settings.persist_directory:
                raise ValueError("Path and persist_directory cannot both be set. path is shorthand for persist_directory={path} and is_persistent=True")
            settings.persist_directory = path if isinstance(path, str) else str(path)
            settings.is_persistent = True
        elif settings.persist_directory and not settings.is_persistent:
            settings.is_persistent = True           

        self.client_kwargs["settings"] = settings
        self._client = self.import_client()(**self.client_kwargs)
        return self._client
   
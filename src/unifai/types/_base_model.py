from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict, RootModel

class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)


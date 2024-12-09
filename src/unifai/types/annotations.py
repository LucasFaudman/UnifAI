from typing import Any, Literal, Union, Sequence, Dict, Collection, Callable, TypeAlias, Type
from .message import Message
from .tool import Tool

from pydantic import BaseModel

ComponentType: TypeAlias = str
ProviderName: TypeAlias = str
ComponentName: TypeAlias = str
ModelName: TypeAlias = str
CollectionName: TypeAlias = str

# Valid input types that can be converted to a Message object
MessageInput: TypeAlias = Message|str|dict

ToolName: TypeAlias = str
# Valid input types that can be converted to a Tool object
ToolInput: TypeAlias = Tool|Type[BaseModel]|Callable|dict[str, Any]|ToolName

# Valid input types that can be converted to a ToolChoice object
ToolChoice: TypeAlias = Literal["auto", "required", "none"]|Tool|ToolName
ToolChoiceInput: TypeAlias = ToolChoice|list[ToolChoice]

# Valid input types that can be converted to a ResponseFormat object
ResponseFormatInput: TypeAlias = Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]

ReturnOnInput: TypeAlias = Literal["content", "tool_call", "message"]|ToolName|Collection[ToolName]

# Valid task types for embeddings. Used to determine what the embeddings are used for to improve the quality of the embeddings
EmbeddingTaskTypeInput: TypeAlias = Literal[
    "retrieval_query", 
    "retrieval_document", 
    "semantic_similarity",
    "classification",
    "clustering",
    "question_answering",
    "fact_verification",
    "code_retrieval_query",
    "image"
]
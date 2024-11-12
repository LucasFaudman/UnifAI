from typing import Any, Literal, Union, Sequence, Collection, Callable, TypeAlias
from .message import Message
from .tool import Tool
from pydantic import BaseModel

# UnifAI Component Types
ComponentType: TypeAlias = Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker", "output_parser", "tool_caller"]

# Supported AI providers
LLMProvider: TypeAlias = Literal["anthropic", "google", "openai", "ollama", "cohere", "nvidia"]

# Supported Embedding providers
EmbeddingProvider: TypeAlias = Literal["google", "openai", "ollama", "cohere", "sentence_transformers", "nvidia"]

# Supported Vector DB providers
VectorDBProvider: TypeAlias = Literal["chroma", "pinecone"]

# Supported Rerank providers
RerankProvider: TypeAlias = Literal["rank_bm25", "cohere", "sentence_transformers", "nvidia"]

# Supported providers
Provider: TypeAlias = LLMProvider|EmbeddingProvider|VectorDBProvider|RerankProvider

# Valid input types that can be converted to a Message object
MessageInput: TypeAlias = Message|str|dict


ToolName: TypeAlias = str
# Valid input types that can be converted to a Tool object
ToolInput: TypeAlias = Tool|BaseModel|Callable|dict[str, Any]|ToolName


# Valid input types that can be converted to a ToolChoice object
ToolChoice: TypeAlias = Literal["auto", "required", "none"]|Tool|ToolName|dict
ToolChoiceInput: TypeAlias = ToolChoice|Sequence[ToolChoice]

# Valid input types that can be converted to a ResponseFormat object
ResponseFormatInput: TypeAlias = Literal["text", "json"]|dict[Literal["json_schema"], dict|Tool|ToolName|BaseModel]

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
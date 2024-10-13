from typing import Any, Literal, Union, Sequence
from .eval_parameters import EvalSpec
from .message import Message
from .tool import Tool

# Supported AI providers
LLMProvider = Literal["anthropic", "google", "openai", "ollama", "cohere", "nvidia"]

# Supported Embedding providers
EmbeddingProvider = Literal["google", "openai", "ollama", "cohere", "sentence_transformers", "nvidia"]

# Supported Vector DB providers
VectorDBProvider = Literal["chroma", "pinecone"]

# Supported Rerank providers
RerankProvider = Literal["rank_bm25", "cohere", "sentence_transformers", "nvidia"]

# Supported providers
Provider = Union[LLMProvider, EmbeddingProvider, VectorDBProvider, RerankProvider] 

# Valid input types that can be converted to a Message object
MessageInput = Union[Message,  dict[str, Any], str]

# Valid input types that can be converted to a Tool object
ToolInput = Union[Tool, dict[str, Any], str]

# Valid input types that can be converted to a ToolChoice object
ToolChoiceInput = Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]
# ToolChoiceInput = Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence["ToolChoiceInput"]]

# Valid input types that can be converted to a ResponseFormat object
ResponseFormatInput = Union[Literal["text", "json", "json_schema"], dict[Literal["type"], Literal["text", "json", "json_schema"]]]

# Valid input types that can be converted to a EvaluateParameters object
EvaluateParametersInput = Union[EvalSpec, dict[str, Any]]

# Valid task types for embeddings. Used to determine what the embeddings are used for to improve the quality of the embeddings
EmbeddingTaskTypeInput = Literal[
    "retreival_query", 
    "retreival_document", 
    "semantic_similarity",
    "classification",
    "clustering",
    "question_answering",
    "fact_verification",
    "code_retreival_query",
    "image"
]
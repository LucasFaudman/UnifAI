from typing import Any, ParamSpec, Literal, Union, Sequence, Dict, Collection, Callable, TypeAlias, TypeVar, Type
from .message import Message

# Input parameters of input_parser/query_modifier/etc callable used to type __call__ of Function, RAGPipe, etc.
InputP = ParamSpec('InputP') 
# Return type of InputParser, query_modifiers, etc which is input to LLM as Message or str->Message(role="user", content=<str_value>)
InputReturnT = TypeVar('InputReturnT', Message, str, Message | str)
# Output type of LLM to be passed to the output_parser callable. 
# str->last_message.content, Message->last_message, ToolCall->last_tool_call, or pass the Chat/Function object (self) to the output_parser callable
OutputT = TypeVar('OutputT')
# OutputT = TypeVar('OutputT', Message, str, ToolCall)
# Return type of the output_parser callable which is the return type of the __call__ method of Function, RAGPipe, etc.
ReturnT = TypeVar('ReturnT')

# Same but used for typing the set_<input/output>_parser methods of Function, RAGPipe, etc.
# func = func.set_input_parser(Callable[NewInputP, NewInputReturnT]) or func.set_output_parser(Callable[[NewOutputT], NewReturnT])
# correctly types func from Function[InputP, InputReturnT, OutputT, ReturnT] to Function[NewInputP, NewInputReturnT, NewOutputT, NewReturnT]
NewInputP = ParamSpec('NewInputP')
NewInputReturnT = TypeVar('NewInputReturnT', Message, str, Message | str)
NewOutputT = TypeVar('NewOutputT')
NewReturnT = TypeVar('NewReturnT')

# Type of default value
DefaultT = TypeVar("DefaultT") 

# Aliases for me so its easy for me to read str purposes debugging and such
ComponentType: TypeAlias = str # "function", "reranker", "vector_db", "tokenizer", etc
ProviderName: TypeAlias = str # "openai", "cohere", "nvidia", etc
ComponentName: TypeAlias = str # unique name for components of same ComponentType and ProviderName ie, "openai-api-key-1", "openai-api-key-2", etc
ModelName: TypeAlias = str # name of LLM, Rereanker, Embedding model, etc ie "llama2-uncensored"
CollectionName: TypeAlias = str # Name of VectorDB, DocumentDB, etc collection ie "manpages", "reviews", etc
ToolName: TypeAlias = str # Name of a Tool ie "get_current_weather", "return_WeatherJoke", etc

# Valid input types that can be converted to a Message object
MessageInput: TypeAlias = Message|str|dict

from .tool import Tool
from pydantic import BaseModel
# Valid input types that can be converted to a Tool object
ToolInput: TypeAlias = Tool|Type[BaseModel]|Callable|dict[str, Any]|ToolName

# Valid input types for tool_choice. 
# auto: automatically choose the best tool based on the input. May return Message with ToolCall or content
# required: required to choose at least 1 tool. Must return Message with ToolCall
# none: do not choose any tool. Must return Message with content not ToolCall
# Tool->ToolName: choose a specific tool by passing the Tool object or its name as a string
ToolChoice: TypeAlias = Literal["auto", "required", "none"]|Tool|ToolName

# Valid tool choices for a chat. A single ToolChoice or a sequence of ToolChoices
# single: set the tool_choice for the chat to the ToolChoice
# sequence: set the tool_choice for the chat to the first ToolChoice in the sequence, after each tool call, the next ToolChoice in the sequence is used
ToolChoiceInput: TypeAlias = ToolChoice|list[ToolChoice]

# Valid input types for response_format.
# text: return the response as a string
# json: return the response as a json object
# json_schema: return the response as a json object that matches the schema dict, BaseModel, or Tool
ResponseFormatInput: TypeAlias = Literal["text", "json"] | dict[Literal["json_schema"], dict[str, str] | Type[BaseModel] | Tool]

# Valid input types for return_on. (When the chat run loop should return and pass output to the OutputParser
# content: return on any Message with content
# tool_call: return on any Message with ToolCall
# message: return on any Message with content or ToolCall
# ToolName: return on any Message with ToolCall for the specified ToolName (usually a return tool ie "return_WeatherJoke")
# Collection[ToolName]: return on any Message with ToolCall for any of the specified ToolNames
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
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator
from typing import overload

from unifai.wrappers._base_client_wrapper import BaseClientWrapper
from unifai.wrappers._base_llm_client import LLMClient
from unifai.wrappers._base_embedding_client import EmbeddingClient


from unifai.wrappers._base_vector_db_client import (
    VectorDBClient, 
    VectorDBIndex,
    VectorDBGetResult,
    VectorDBQueryResult
)

from unifai.types import (
    AIProvider,
    VectorDBProvider,
    Provider, 
    EvaluateParameters,
    EvaluateParametersInput, 
    Message,
    MessageChunk,
    MessageInput, 
    Tool,
    ToolInput,
    Embeddings,
    Embedding
)
from unifai.type_conversions import make_few_shot_prompt, standardize_eval_prameters, standardize_tools
from .chat import Chat
# from .chroma_emebedding import UnifAIChromaEmbeddingFunction, get_chroma_client
from pathlib import Path

AI_PROVIDERS: frozenset[AIProvider] = frozenset(("anthropic", "google", "openai", "ollama"))
VECTOR_DB_PROVIDERS: frozenset[VectorDBProvider] = frozenset(("chroma", "pinecone"))
REQUIRES_PARENT: frozenset[Provider] = frozenset(("chroma", "pinecone"))

class UnifAIClient:
    TOOLS: list[ToolInput] = []
    TOOL_CALLABLES: dict[str, Callable] = {}
    EVAL_PARAMETERS: list[EvaluateParametersInput] = []





    
    def __init__(self, 
                 provider_client_kwargs: Optional[dict[Provider, dict[str, Any]]] = None,
                 tools: Optional[list[ToolInput]] = None,
                 tool_callables: Optional[dict[str, Callable]] = None,
                 eval_prameters: Optional[list[EvaluateParametersInput]] = None,
                 default_ai_provider: Optional[AIProvider] = None,
                 default_vector_db_provider: Optional[VectorDBProvider] = None,                 
                 ):
        
        # self.provider_client_kwargs = provider_client_kwargs if provider_client_kwargs is not None else {}        
        # self.providers = list(self.provider_client_kwargs.keys())        
        # self.ai_providers = [provider for provider in self.providers if provider in AI_PROVIDERS]
        # self.vector_db_providers = [provider for provider in self.providers if provider in VECTOR_DB_PROVIDERS]
        # self.default_ai_provider: Provider = self.providers[0] if len(self.providers) > 0 else "openai"

        self.set_provider_client_kwargs(provider_client_kwargs)
        self.set_default_ai_provider(default_ai_provider)
        self.set_default_vector_db_provider(default_vector_db_provider)
        
        self._clients: dict[Provider, LLMClient|VectorDBClient] = {}
        self.tools: dict[str, Tool] = {}
        self.tool_callables: dict[str, Callable] = {}
        self.eval_prameters: dict[str, EvaluateParameters] = {}
        
        self.add_tools(tools or self.TOOLS)
        self.add_tool_callables(tool_callables)
        self.add_eval_prameters(eval_prameters or self.EVAL_PARAMETERS)
        

    def set_provider_client_kwargs(self, provider_client_kwargs: Optional[dict[Provider, dict[str, Any]]] = None):
        self.provider_client_kwargs = provider_client_kwargs if provider_client_kwargs is not None else {}        
        self.providers: list[Provider] = list(self.provider_client_kwargs.keys())
        self.ai_providers: list[AIProvider] = [provider for provider in self.providers if provider in AI_PROVIDERS]
        self.vector_db_providers: list[VectorDBProvider] = [provider for provider in self.providers if provider in VECTOR_DB_PROVIDERS]        

    def set_default_ai_provider(self, provider: Optional[AIProvider] = None, check: bool = True):
        if check and provider and provider not in AI_PROVIDERS:
            raise ValueError(f"Invalid AI provider: {provider}. Must be one of: {AI_PROVIDERS}")
        if provider:
            self.default_ai_provider: AIProvider = provider
        elif self.ai_providers:
            self.default_ai_provider = self.ai_providers[0]
        else:
            self.default_ai_provider = "openai"

    def set_default_vector_db_provider(self, provider: Optional[VectorDBProvider] = None, check: bool = True):
        if check and provider and provider not in VECTOR_DB_PROVIDERS:
            raise ValueError(f"Invalid Vector DB provider: {provider}. Must be one of: {VECTOR_DB_PROVIDERS}")
        if provider:
            self.default_vector_db_provider: VectorDBProvider = provider
        elif self.vector_db_providers:
            self.default_vector_db_provider = self.vector_db_providers[0]
        else:
            self.default_vector_db_provider = "chroma"


    def add_tools(self, tools: Optional[list[ToolInput]]):
        if not tools: return

        for tool_name, tool in standardize_tools(tools).items():
            self.tools[tool_name] = tool
            if (tool_callable := getattr(tool, "callable", None)) is not None:
                self.tool_callables[tool_name] = tool_callable

    def add_tool_callables(self, tool_callables: Optional[dict[str, Callable]]):
        if not tool_callables: return
        self.tool_callables.update(tool_callables)


    def add_eval_prameters(self, eval_prameters: Optional[list[EvaluateParametersInput]]):
        if not eval_prameters: return
        self.eval_prameters.update(standardize_eval_prameters(eval_prameters))


    def import_client_wrapper(self, provider: Provider) -> Type[LLMClient|VectorDBClient]:
        match provider:
            # AI Client Wrappers
            case "anthropic":
                from unifai.wrappers.anthropic import AnthropicWrapper
                return AnthropicWrapper
            case "google":
                from unifai.wrappers.google import GoogleAIWrapper
                return GoogleAIWrapper
            case "openai":
                from unifai.wrappers.openai import OpenAIWrapper
                return OpenAIWrapper
            case "ollama":
                from unifai.wrappers.ollama import OllamaWrapper
                return OllamaWrapper
            # Embedding Vector DB Client Wrappers
            case "chroma":
                from unifai.wrappers.chroma import ChromaClient
                return ChromaClient
            case "pinecone":
                from unifai.wrappers.pinecone import PineconeClient
                return PineconeClient
            case _:
                raise ValueError(f"Invalid provider: {provider}")
            

    def init_client(self, provider: Provider, *client_args, **client_kwargs) -> LLMClient|VectorDBClient:
        client_kwargs = {**self.provider_client_kwargs[provider], **client_kwargs}
        if provider in REQUIRES_PARENT and "parent" not in client_kwargs:
            client_kwargs["parent"] = self
        client_kwarg_args = client_kwargs.pop("args", ())
        self._clients[provider] = self.import_client_wrapper(provider)(*client_args, *client_kwarg_args, **client_kwargs)
        return self._clients[provider]
    

    @overload
    def get_client(self, provider: AIProvider, **client_kwargs) -> LLMClient:
        ...

    @overload
    def get_client(self, provider: VectorDBProvider, **client_kwargs) -> VectorDBClient:
        ...        

    def get_client(self, provider: Provider, **client_kwargs) -> LLMClient|VectorDBClient:
        provider = provider or self.default_ai_provider
        if provider not in self._clients or (client_kwargs and self._clients[provider].client_kwargs != client_kwargs):
            return self.init_client(provider, **client_kwargs)
        return self._clients[provider]

    def get_ai_client(self, provider: Optional[AIProvider] = None, **client_kwargs) -> LLMClient:
        provider = provider or self.default_ai_provider
        return self.get_client(provider, **client_kwargs)

    def get_vector_db_client(self, provider: Optional[VectorDBProvider] = None, **client_kwargs) -> VectorDBClient:
        provider = provider or self.default_vector_db_provider
        return self.get_client(provider, **client_kwargs)

    # List Models
    def list_models(self, provider: Optional[AIProvider] = None) -> list[str]:
        return self.get_ai_client(provider).list_models()


    def start_chat(
            self,
            **kwargs
            # messages: Optional[Sequence[MessageInput]] = None,
            # provider: Optional[AIProvider] = None,            
            # model: Optional[str] = None,
            # system_prompt: Optional[str] = None,             
            # tools: Optional[Sequence[ToolInput]] = None,
            # tool_callables: Optional[dict[str, Callable]] = None,
            # tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            # response_format: Optional[Union[str, dict[str, str]]] = None,

            # return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            # enforce_tool_choice: bool = False,
            # tool_choice_error_retries: int = 3,

            # max_tokens: Optional[int] = None,
            # frequency_penalty: Optional[float] = None,
            # presence_penalty: Optional[float] = None,
            # seed: Optional[int] = None,
            # stop_sequences: Optional[list[str]] = None, 
            # temperature: Optional[float] = None,
            # top_k: Optional[int] = None,
            # top_p: Optional[float] = None,
            ) -> Chat:
            return Chat(
                parent=self,
                provider=kwargs.pop("provider", self.default_ai_provider),
                messages=kwargs.pop("messages", []),
                **kwargs
                # model=model,
                # system_prompt=system_prompt,
                # tools=tools,
                # tool_callables=tool_callables,
                # tool_choice=tool_choice,
                # response_format=response_format,
                # return_on=return_on,
                # enforce_tool_choice=enforce_tool_choice,
                # tool_choice_error_retries=tool_choice_error_retries,

                # max_tokens=max_tokens,
                # frequency_penalty=frequency_penalty,
                # presence_penalty=presence_penalty,
                # seed=seed,
                # stop_sequences=stop_sequences,
                # temperature=temperature,
                # top_k=top_k,
                # top_p=top_p,                
            )

    
    def chat(
            self,
            messages: Optional[Sequence[MessageInput]] = None,
            provider: Optional[AIProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,             
            tools: Optional[Sequence[ToolInput]] = None,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,

            return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            enforce_tool_choice: bool = True,
            tool_choice_error_retries: int = 3,

            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,

            **kwargs
            ) -> Chat:
            chat = self.start_chat(
                messages=messages if messages is not None else [],
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                tool_callables=tool_callables,
                tool_choice=tool_choice,
                response_format=response_format,

                return_on=return_on,
                enforce_tool_choice=enforce_tool_choice,
                tool_choice_error_retries=tool_choice_error_retries,

                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                stop_sequences=stop_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if messages:
                chat.run(**kwargs)
            return chat
        

    def chat_stream(
            self,
            messages: Optional[Sequence[MessageInput]] = None,
            provider: Optional[AIProvider] = None,            
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,             
            tools: Optional[Sequence[ToolInput]] = None,
            tool_callables: Optional[dict[str, Callable]] = None,
            tool_choice: Optional[Union[Literal["auto", "required", "none"], Tool, str, dict, Sequence[Union[Tool, str, dict]]]] = None,
            response_format: Optional[Union[str, dict[str, str]]] = None,

            return_on: Union[Literal["content", "tool_call", "message"], str, Collection[str]] = "content",
            enforce_tool_choice: bool = True,
            tool_choice_error_retries: int = 3,

            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,

            **kwargs
            ) -> Generator[MessageChunk, None, Chat]:

            chat = self.start_chat(
                messages=messages,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                tool_callables=tool_callables,
                tool_choice=tool_choice,
                response_format=response_format,

                return_on=return_on,
                enforce_tool_choice=enforce_tool_choice,
                tool_choice_error_retries=tool_choice_error_retries,

                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                stop_sequences=stop_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if messages:
                yield from chat.run_stream(**kwargs)
            return chat
    

    def evaluate(self, 
                 eval_type: str | EvaluateParameters, 
                 content: Any, 
                 provider: Optional[AIProvider] = None,
                 model: Optional[str] = None,
                 **kwargs
                 ) -> Any:
        
        # Get eval_parameters from eval_type
        if isinstance(eval_type, str):
            if (eval_parameters := self.eval_prameters.get(eval_type)) is None:
                raise ValueError(f"Eval type '{eval_type}' not found in eval_prameters")
        elif isinstance(eval_type, EvaluateParameters):
            eval_parameters = eval_type
        else:
            raise ValueError(
                f"Invalid eval_type: {eval_type}. Must be a string (eval_type of EvaluateParameters in self.EVAL_PARAMETERS) or an EvaluateParameters object")

        # Determine return_on parameter from eval_parameters and tool_choice
        if eval_parameters.return_on:
            # Use the return_on parameter from eval_parameters if provided
            return_on = eval_parameters.return_on
        elif isinstance(eval_parameters.tool_choice, str):
            # Use the tool_choice parameter from eval_parameters if its a string (single tool name)
            return_on = eval_parameters.tool_choice
        elif eval_parameters.tool_choice:
            # Use the last tool choice if tool_choice is a non-empty sequence of tool names (tool_choice queue)
            return_on = eval_parameters.tool_choice[-1]
        else:
            # Default to return on content if no return_on or tool_choice is provided
            return_on = "content"

        # Create input messages from system_prompt, few-shot examples, and content
        input_messages = make_few_shot_prompt(
            system_prompt=eval_parameters.system_prompt,
            examples=eval_parameters.examples,
            content=content
        )

        # Initialize and run chat
        chat = self.chat(
            messages=input_messages,
            provider=provider,
            model=model,
            tools=eval_parameters.tools,
            tool_choice=eval_parameters.tool_choice,
            response_format=eval_parameters.response_format,
            return_on=return_on,
            **kwargs
        )
        
        # Return the desired attribute of the chat object or the chat object itself based on eval_parameters.return_as
        return getattr(chat, eval_parameters.return_as) if eval_parameters.return_as != "chat" else chat
    


    def embed(self, 
              input: str | Sequence[str],
              model: Optional[str] = None,
              provider: Optional[AIProvider] = None,
              max_dimensions: Optional[int] = None,
              **kwargs
              ) -> Embeddings:
        
        if max_dimensions is not None and max_dimensions < 1:
            raise ValueError(f"Embedding max_dimensions must be greater than 0. Got: {max_dimensions}")
        return self.get_ai_client(provider).embed(input, model, max_dimensions, **kwargs)



    def get_or_create_index(self, 
                            name: str,
                            vector_db_provider: Optional[VectorDBProvider] = None,                            
                            embedding_provider: Optional[AIProvider] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                            index_metadata: Optional[dict] = None,
                            **kwargs
                            ) -> VectorDBIndex:
        return self.get_vector_db_client(vector_db_provider).get_or_create_index(
            name=name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            metadata=index_metadata,
            **kwargs
        )
    

    def upsert_index(self,
                     name: str,
                     ids: list[str],
                     metadatas: Optional[list[dict]] = None,
                     documents: Optional[list[str]] = None,
                     embeddings: Optional[list[Embedding]] = None,
                     vector_db_provider: Optional[VectorDBProvider] = None,                            
                     embedding_provider: Optional[AIProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                     index_metadata: Optional[dict] = None
                     ) -> VectorDBIndex:
        return self.get_or_create_index(
              name=name,
              vector_db_provider=vector_db_provider,
              embedding_provider=embedding_provider,
              embedding_model=embedding_model,
              dimensions=dimensions,
              distance_metric=distance_metric,
              index_metadata=index_metadata
            ).upsert(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings
            )
    

    def query_index(self, 
                    name: str,
                    query: str | list[str] | Embedding | list[Embedding] | Embeddings,
                    n_results: int = 10,
                    where: Optional[dict] = None,
                    where_document: Optional[dict] = None,
                    include: Sequence[Literal["metadatas", "documents", "distances"]] = ("metadatas", "documents", "distances"),
                    vector_db_provider: Optional[VectorDBProvider] = None,                            
                    embedding_provider: Optional[AIProvider] = None,
                    embedding_model: Optional[str] = None,
                    dimensions: Optional[int] = None,
                    distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                    index_metadata: Optional[dict] = None                    
              ) -> VectorDBQueryResult:                     
        
        query_texts = None
        query_embeddings = None                            
        if isinstance(query, str):
            query_texts = [query] # Single string
        elif isinstance(query, list):
            item_0 = query[0]
            if isinstance(item_0, str):
                query_texts = query # List of strings
            elif isinstance(item_0, float):
                query_embeddings = [query] # Single Embedding (list of floats)
            elif isinstance(item_0, list) and isinstance(item_0[0], float):
                query_embeddings = query # List of Embeddings (list of lists of floats)
        
        if not query_texts and not query_embeddings:
            raise ValueError(f"Invalid query type: {type(query)}. Must be a str, list of str, Embedding (list[float]), list of Embedding list[list[float]] or Embeddings object returned by embed()")

                    
        return self.get_or_create_index(
                name=name,
                vector_db_provider=vector_db_provider,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                dimensions=dimensions,
                distance_metric=distance_metric,
                index_metadata=index_metadata
                ).query(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                    include=include
                )
from typing import Optional, Union, Sequence, Any, Literal, Mapping, Iterator, Generator

import google.generativeai as genai
from google.generativeai import (
    configure as genai_configure,
    embed_content as genai_embed_content,
    list_models as genai_list_models,
    get_model as genai_get_model,
    ChatSession,
    GenerationConfig,
    GenerativeModel,
)
from google.generativeai.types.content_types import (
    to_tool_config,
    to_content,
    to_contents,
    to_blob,
    to_part,

)
from google.generativeai.types import (
    GenerateContentResponse,
    ContentType,
    BlockedPromptException,
    StopCandidateException,
    IncompleteIterationError,
    BrokenResponseError,
)   
    

# from google.generativeai.discuss import (
#     chat as genai_chat,
#     ChatResponse,
# )
from google.generativeai.protos import (
    Blob,
    CodeExecution,
    CodeExecutionResult,
    FunctionCall,
    FunctionCallingConfig,
    FunctionDeclaration,
    FunctionResponse,
    Part,
    Schema,
    Tool as GoogleTool,
    ToolConfig,
    Candidate,
    Content,
)

from google.api_core.exceptions import (
    GoogleAPICallError,
)


from unifai.exceptions import (
    UnifAIError,
    APIError,
    UnknownAPIError,
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    STATUS_CODE_TO_EXCEPTION_MAP,   
    ProviderUnsupportedFeatureError,
    ModelUnsupportedFeatureError
)
from google.protobuf.struct_pb2 import Struct as GoogleProtobufStruct

from unifai.types import (
    Message, 
    MessageChunk,
    Tool, 
    ToolCall, 
    Image, 
    Usage, 
    ResponseInfo, 
    ToolParameter,
    StringToolParameter,
    NumberToolParameter,
    IntegerToolParameter,
    BooleanToolParameter,
    NullToolParameter,
    ArrayToolParameter,
    ObjectToolParameter,
    AnyOfToolParameter,
    OptionalToolParameter,
    RefToolParameter,
    Embeddings,
)
from unifai.type_conversions import stringify_content
from .._core._base_llm_client import LLMClient
from .._core._base_embedder import Embedder, EmbeddingTaskTypeInput

from random import choices as random_choices
from string import ascii_letters, digits
def generate_random_id(length=8):
    return ''.join(random_choices(ascii_letters + digits, k=length))

GoogleEmbeddingTaskType = Literal[
    "RETRIEVAL_QUERY",      # Specifies the given text is a query in a search or retrieval setting.
    "RETRIEVAL_DOCUMENT",   # Specifies the given text is a document in a search or retrieval setting.
    "SEMANTIC_SIMILARITY",  # Specifies the given text is used for Semantic Textual Similarity (STS).
    "CLASSIFICATION",       # Specifies that the embedding is used for classification.
    "CLUSTERING",	        # Specifies that the embedding is used for clustering.
    "QUESTION_ANSWERING",	# Specifies that the query embedding is used for answering questions. Use RETRIEVAL_DOCUMENT for the document side.
    "FACT_VERIFICATION",	# Specifies that the query embedding is used for fact verification.
    "CODE_RETRIEVAL_QUERY"  # Specifies that the query embedding is used for code retrieval for Java and Python. 
]

class GoogleAIAdapter(Embedder, LLMClient):
    provider = "google"
    default_model = "gemini-1.5-flash-latest"
    default_embedding_model = "text-embedding-004"
    
    model_embedding_dimensions = {
        "textembedding-gecko@001": 768,
        "textembedding-gecko-multilingual@001": 768,
        "textembedding-gecko@003": 768,
        "text-multilingual-embedding-002": 768,
        "text-embedding-004	": 768,
        "text-embedding-preview-0815": 768,
    }
    # models_with_embedding_task_type_support = {
    #     "models/embedding-001",
    # }

    def import_client(self):
        import google.generativeai as genai
        return genai


    def init_client(self, **client_kwargs) -> Any:
        self._client = self.import_client()
        self._client.configure(**client_kwargs)
        return self._client


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: Exception) -> UnifAIError:      
        if isinstance(exception, GoogleAPICallError):
            message = exception.message
            status_code = exception.code

            if status_code == 400:
                if "API key" in message:                    
                    status_code = 401 # BadRequestError to AuthenticationError
                elif "unexpected model" in message:
                    status_code = 404 # BadRequestError to NotFoundError
            elif status_code == 403 and "authentication" in message:
                status_code = 401 # PermissionDeniedError to AuthenticationError                                      
        else:
            message = str(exception)
            status_code = None

        if status_code is not None:
            unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
            unifai_exception_type = UnknownAPIError
        return unifai_exception_type(message=message, status_code=status_code, original_exception=exception)


    def format_model_name(self, model: str) -> str:
        if model.startswith("models/"):
            return model
        return f"models/{model}"


    # List Models
    def list_models(self) -> list[str]:
        return [model.name[7:] for model in self.client.list_models()]


    # Embeddings
    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput] = None,
                            task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported"
                            ) -> Optional[GoogleEmbeddingTaskType]:
        if task_type is None:
            return None
        if "@001" in model:
            raise ModelUnsupportedFeatureError(
                f"Model {model} does not support task_type specification for embeddings. "
                "See: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types"
            )

        if (_task_type := task_type.upper()) != "IMAGE":
            return _task_type # GoogleAI supports all input types except "image" # type: ignore
        if task_type_not_supported == "use_closest_supported":
            return "RETRIEVAL_DOCUMENT"
        raise ProviderUnsupportedFeatureError(
            f"Embedding task_type={task_type} is not supported by Google. "
             "Supported input types are 'retreival_query', 'retreival_document', "
             "'semantic_similarity', 'classification', 'clustering', 'question_answering', "
             "'fact_verification', and 'code_retreival_query'. Use 'retreival_document' for images with Google. "
             "Or use Cohere which supports embedding 'image' task_type."
        )


    def _get_embed_response(
            self,            
            input: Sequence[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Optional[GoogleEmbeddingTaskType] = None,
            input_too_large: Literal[
                "truncate_end", 
                "truncate_start", 
                "raise_error"
                ] = "truncate_end",
            **kwargs
            ) -> Any:
        
        return self.client.embed_content(
            content=input,
            model=self.format_model_name(model),
            output_dimensionality=dimensions,
            task_type=task_type,
            # TODO - Preform this logic only when using Vertex AI models
            # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#parameter-list
            # auto_truncate=(input_too_large != "raise_error"),
            **kwargs
        )
    

    def _extract_embeddings(
            self,            
            response: Mapping,
            model: str,
            **kwargs
            ) -> Embeddings:
        
        return Embeddings(
            root=response["embedding"],
            response_info=ResponseInfo(
                model=model, 
                usage=Usage()
            )
        ) 


    # Chat
    def _get_chat_response(
            self,
            stream: bool,
            messages: list[Content],     
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,                   
            tools: Optional[list[Any]] = None,
            tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,            
            response_format: Optional[Union[str, dict[str, str]]] = None,
            max_tokens: Optional[int] = None,
            frequency_penalty: Optional[float] = None,
            presence_penalty: Optional[float] = None,
            seed: Optional[int] = None,
            stop_sequences: Optional[list[str]] = None, 
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None, 
            **kwargs
            ) -> tuple[Message, Any]:
        
        model = self.format_model_name(model) if model else self.default_model
        gen_config = GenerationConfig(
            # candidate_count=1,
            stop_sequences=stop_sequences,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            response_mime_type=response_format, #text/plain or application/json
        )

        gen_model = self.client.GenerativeModel(
            model_name=model,
            safety_settings=kwargs.pop("safety_settings", None),
            generation_config=gen_config,
            tools=tools,
            tool_config=tool_choice,
            system_instruction=system_prompt,            
        )
        # genai.GenerativeModel.generate_content
        return gen_model.generate_content(
            messages,
            stream=stream,
            **kwargs
        )


    # Convert Objects from UnifAI to AI Provider format        
        # Messages    
    def format_user_message(self, message: Message) -> Any:
        parts = []
        if message.content is not None:
            parts.append(Part(text=message.content))
        if message.images:
            parts.extend(map(self.format_image, message.images))
        return {"role": "user", "parts": parts}


    def format_assistant_message(self, message: Message) -> Any:
        parts = []
        if message.content is not None:
            parts.append(Part(text=message.content))
        if message.tool_calls:
            for tool_call in message.tool_calls:
                args_struct = GoogleProtobufStruct()
                args_struct.update(tool_call.arguments)
                parts.append(
                    FunctionCall(
                        name=tool_call.tool_name,
                        args=args_struct,
                    )
                )
        if message.images:
            parts.extend(map(self.format_image, message.images))
        if not parts:
            # Should not happen unless switching providers after a tool call
            parts.append(Part(text="continue"))       
        return {"role": "model", "parts": parts}
        

    def format_tool_message(self, message: Message) -> Any:
        parts = []
        # if message.content:
        #     parts.append(Part(text=message.content))
        if message.tool_calls:
            for tool_call in message.tool_calls:
                result_struct = GoogleProtobufStruct()
                result_struct.update({"result": tool_call.output})
                parts.append(
                    FunctionResponse(
                        name=tool_call.tool_name,
                        response=result_struct,
                    )
                )
        if message.images:
            parts.extend(map(self.format_image, message.images))
        return {"role": "user", "parts": parts}
    
    
    def format_system_message(self, message: Message) -> Any:
        raise ValueError("GoogleAI does not support system messages")
    

    def format_messages_and_system_prompt(self, 
                                              messages: list[Message], 
                                              system_prompt_arg: Optional[str] = None
                                              ) -> tuple[list, Optional[str]]:
        
        system_prompt = system_prompt_arg
        if messages and messages[0].role == "system":
            # Remove the first system message from the list since Anthropic does not support system messages
            system_message = messages.pop(0)
            # Set the system prompt to the content of the first system message if not set by the argument
            if not system_prompt:
                system_prompt = system_message.content 

        client_messages = [self.format_message(message) for message in messages]
        return client_messages, system_prompt   


        # Images
    def format_image(self, image: Image) -> Any:
        return Blob(data=image.raw_bytes, mime_type=image.mime_type)
    

        # Tools
    def format_tool(self, tool: Tool) -> GoogleTool:

        def tool_parameter_to_schema(tool_parameter: ToolParameter) -> Schema:        
            if isinstance(tool_parameter, OptionalToolParameter):
                tool_parameter = tool_parameter.anyOf[0]
                nullable = True
            else:
                nullable = False            
            if isinstance(tool_parameter, (AnyOfToolParameter, RefToolParameter)):
                raise ValueError(f"{tool_parameter.__class__.__name__} is not supported by GoogleAI")

            items = None
            properties = None
            required = None           
            if isinstance(tool_parameter, ObjectToolParameter):
                properties = {}
                required = []
                for prop in tool_parameter.properties:
                    properties[prop.name] = tool_parameter_to_schema(prop)
                    required.append(prop.name)
            elif isinstance(tool_parameter, ArrayToolParameter):
                items = tool_parameter_to_schema(tool_parameter.items)                         

            return Schema(
                type=tool_parameter.type.upper(),
                # format=tool_parameter.format,
                description=tool_parameter.description,
                nullable=nullable, 
                enum=tool_parameter.enum,
                # maxItems=tool_parameter.maxItems,
                properties=properties,
                required=required,
                items=items
            )
        
        function_declaration = FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=tool_parameter_to_schema(tool.parameters) if tool.parameters else None,
        )
        return GoogleTool(function_declarations=[function_declaration])
    
        
    def format_tool_choice(self, tool_choice: str) -> ToolConfig:
        if tool_choice in ("auto", "required", "none"):
            mode = tool_choice.upper() if tool_choice != "required" else "ANY"
            allowed_function_names = []
        else:
            mode = "ANY"
            allowed_function_names = [tool_choice,]
        
        return ToolConfig(
            function_calling_config={
                "mode": mode,
                "allowed_function_names": allowed_function_names
            }
        )


        # Response Format
    def format_response_format(self, response_format: str) -> str:
        if 'json' in response_format:
            return "application/json"
        elif 'text' in response_format:
            return "text/plain"
        return response_format


    # Convert Objects from AI Provider to UnifAI format    
        # Images
    def parse_image(self, response_image: Any) -> Image:
        raise NotImplementedError("This method must be implemented by the subclass")


        # Tool Calls
    def parse_tool_call(self, response_tool_call: FunctionCall, **kwargs) -> ToolCall:
            return ToolCall(
                id=f'call_{generate_random_id(24)}',
                tool_name=response_tool_call.name,
                arguments=dict(response_tool_call.args)
        )
    

        # Response Info (Model, Usage, Done Reason, etc.)
    def parse_done_reason(self, response_obj: Any, **kwargs) -> str|None:
        done_reason = response_obj.finish_reason
        if not done_reason:
            return None
        elif done_reason == 1:
            return "tool_calls" if kwargs.get("tools_called") else "stop"
        elif done_reason == 2:
            return "max_tokens"
        elif done_reason in (5, 10):
            return "error"
        else:
            return "content_filter"
    

    def parse_usage(self, response_obj: GenerateContentResponse, **kwargs) -> Usage|None:
        if response_obj.usage_metadata:
            return Usage(
                input_tokens=response_obj.usage_metadata.prompt_token_count,
                output_tokens=response_obj.usage_metadata.cached_content_token_count,
            )


    def parse_response_info(self, response: GenerateContentResponse, **kwargs) -> ResponseInfo:        
        return ResponseInfo(
            model=kwargs.get("model"), 
            done_reason=self.parse_done_reason(response.candidates[0], **kwargs), 
            usage=self.parse_usage(response)
        )


    def _extract_parts(self, parts: Sequence[Part]) -> tuple[str|None, list[ToolCall]|None, list[Image]|None]:   
        # content = None
        content = ""
        tool_calls = None
        images = None
        for part in parts:
            if part.text:
                # if content is None:
                #     content = ""
                content += part.text
            elif part.function_call:
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(self.parse_tool_call(part.function_call))
            elif part.inline_data:
                if images is None:
                    images = []
                images.append(self.parse_image(part.inline_data))
            elif part.file_data or part.executable_code or part.code_execution_result:
                raise NotImplementedError("file_data, executable_code, and code_execution_result are not yet supported by UnifAI")
        return content, tool_calls, images


        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def parse_message(self, response: GenerateContentResponse, **kwargs) -> tuple[Message, Content]:
        client_message = response.candidates[0].content
        content, tool_calls, images = self._extract_parts(client_message.parts)
        response_info = self.parse_response_info(response, tools_called=bool(tool_calls), **kwargs)
        std_message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            images=images,
            response_info=response_info
        )        
        return std_message, client_message

    
    def parse_stream(self, response: GenerateContentResponse, **kwargs) -> Generator[MessageChunk, None, tuple[Message, Content]]:
        parts = []
        for chunk in response:
            chunk_parts = list(chunk.parts)
            parts.extend(chunk_parts)
            content, tool_calls, images = self._extract_parts(chunk_parts)
        
            yield MessageChunk(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                images=images
            )

        # response.parts.extend(parts)
        response.candidates[0].content.parts = parts
        return self.parse_message(response, **kwargs)



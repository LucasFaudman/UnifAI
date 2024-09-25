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
    STATUS_CODE_TO_EXCEPTION_MAP   
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
    Embedding,
    EmbedResult,
)
from unifai.type_conversions import stringify_content
from ._base import BaseAIClientWrapper

from random import choices as random_choices
from string import ascii_letters, digits
def generate_random_id(length=8):
    return ''.join(random_choices(ascii_letters + digits, k=length))

class GoogleAIWrapper(BaseAIClientWrapper):
    provider = "google"
    default_model = "gemini-1.5-flash-latest"
    default_embedding_model = "text-embedding-004"

    def import_client(self):
        import google.generativeai as genai
        return genai

    def init_client(self, **client_kwargs) -> Any:
        self._client = self.import_client()
        self._client.configure(**client_kwargs)
        return self._client


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: Exception) -> UnifAIError:
        # if isinstance(exception, (BrokenResponseError, IncompleteIterationError)):
        
        if isinstance(exception, GoogleAPICallError):
            message = exception.message
            status_code = exception.code

            if status_code == 400:
                if "API key" in message:
                    # Convert BadRequestError to AuthenticationError
                    status_code = 401
                elif "unexpected model" in message:
                    # Convert BadRequestError to NotFoundError
                    status_code = 404

            elif status_code == 403 and "authentication" in message:
                # Convert PermissionDeniedError to AuthenticationError
                status_code = 401
                                    
        else:
            message = str(exception)
            status_code = None

        if status_code is not None:
            unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
            unifai_exception_type = UnknownAPIError

        return unifai_exception_type(message=message, status_code=status_code, original_exception=exception)

        
    # Convert Objects from UnifAI to AI Provider format        
        # Messages    
    def prep_input_user_message(self, message: Message) -> Any:
        parts = []
        if message.content:
            parts.append(Part(text=message.content))
        if message.images:
            parts.extend(map(self.prep_input_image, message.images))
        return {"role": "user", "parts": parts}

    def prep_input_assistant_message(self, message: Message) -> Any:
        parts = []
        if message.content:
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
            parts.extend(map(self.prep_input_image, message.images))
        return {"role": "model", "parts": parts}
        
    def prep_input_tool_message(self, message: Message) -> Any:
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
            parts.extend(map(self.prep_input_image, message.images))
        return {"role": "user", "parts": parts}
    
    
    def prep_input_system_message(self, message: Message) -> Any:
        raise ValueError("GoogleAI does not support system messages")
    

    def prep_input_messages_and_system_prompt(self, 
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

        client_messages = [self.prep_input_message(message) for message in messages]
        return client_messages, system_prompt   
       
        # Images
    def prep_input_image(self, image: Image) -> Any:
        return Blob(data=image.raw_bytes, mime_type=image.mime_type)
    

        # Tools
    def prep_input_tool(self, tool: Tool) -> GoogleTool:

        def tool_parameter_to_schema(tool_parameter: ToolParameter) -> Schema:        
            if isinstance(tool_parameter, AnyOfToolParameter):
                raise ValueError("AnyOfToolParameter is not supported by GoogleAI")

            items = None
            properties = None
            required = None           
            if isinstance(tool_parameter, ObjectToolParameter):
                properties = {}
                required = []
                for prop in tool_parameter.properties:
                    properties[prop.name] = tool_parameter_to_schema(prop)
                    if prop.required:
                        required.append(prop.name)
            elif isinstance(tool_parameter, ArrayToolParameter):
                items = tool_parameter_to_schema(tool_parameter.items)                         

            return Schema(
                type=tool_parameter.type.upper(),
                # format=tool_parameter.format,
                description=tool_parameter.description,
                nullable=True if not tool_parameter.required else None, # not tool_parameter.required?
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
    
        
    def prep_input_tool_choice(self, tool_choice: str) -> ToolConfig:
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
    def prep_input_response_format(self, response_format: str) -> str:
        if 'json' in response_format:
            return "application/json"
        elif 'text' in response_format:
            return "text/plain"
        return response_format


    # Convert Objects from AI Provider to UnifAI format    
        # Images
    def extract_image(self, response_image: Any) -> Image:
        raise NotImplementedError("This method must be implemented by the subclass")

        # Tool Calls
    def extract_tool_call(self, response_tool_call: FunctionCall, **kwargs) -> ToolCall:
            return ToolCall(
                id=f'call_{generate_random_id(24)}',
                tool_name=response_tool_call.name,
                arguments=dict(response_tool_call.args)
        )
    
        # Response Info (Model, Usage, Done Reason, etc.)
    def extract_done_reason(self, response_obj: Any, **kwargs) -> str|None:
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
    
    def extract_usage(self, response_obj: GenerateContentResponse, **kwargs) -> Usage|None:
        if response_obj.usage_metadata:
            return Usage(
                input_tokens=response_obj.usage_metadata.prompt_token_count,
                output_tokens=response_obj.usage_metadata.cached_content_token_count,
            )

    def extract_response_info(self, response: GenerateContentResponse, **kwargs) -> ResponseInfo:
        
        # finish_reason = response.candidates[0].finish_reason
        # if not finish_reason:
        #     done_reason = None
        # elif finish_reason == 1:
        #     done_reason = "stop" if not tools_called else "tool_calls"
        # elif finish_reason == 2:
        #     done_reason = "max_tokens"
        # elif finish_reason in (5, 10):
        #     done_reason = "error"
        # else:
        #     done_reason = "content_filter"
        
        done_reason = self.extract_done_reason(response.candidates[0], **kwargs)
        # usage = Usage(
        #         input_tokens=response.usage_metadata.prompt_token_count,
        #         output_tokens=response.usage_metadata.cached_content_token_count,
        #     )
        usage = self.extract_usage(response)
        return ResponseInfo(model=kwargs.get("model"), done_reason=done_reason, usage=usage)

    def _extract_parts(self, parts: Sequence[Part]) -> tuple[str|None, list[ToolCall]|None, list[Image]|None]:   
        content = None
        tool_calls = None
        images = None
        for part in parts:
            if part.text:
                if content is None:
                    content = ""
                content += part.text
            elif part.function_call:
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(self.extract_tool_call(part.function_call))
            elif part.inline_data:
                if images is None:
                    images = []
                images.append(self.extract_image(part.inline_data))
            elif part.file_data or part.executable_code or part.code_execution_result:
                raise NotImplementedError("file_data, executable_code, and code_execution_result are not yet supported by UnifAI")
        return content, tool_calls, images

        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def extract_assistant_message_both_formats(self, response: GenerateContentResponse, **kwargs) -> tuple[Message, Content]:
        client_message = response.candidates[0].content
        # content = ""
        # tool_calls = []
        # images = []
        # for part in client_message.parts:
        #     if part.text:
        #         content += part.text
        #     elif part.function_call:
        #         tool_calls.append(self.extract_tool_call(part.function_call))
        #     elif part.inline_data:
        #         images.append(self.extract_image(part.inline_data))
        #     elif part.file_data or part.executable_code or part.code_execution_result:
        #         raise NotImplementedError("file_data, executable_code, and code_execution_result are not yet supported by UnifAI")

        content, tool_calls, images = self._extract_parts(client_message.parts)

        response_info = self.extract_response_info(response, tools_called=bool(tool_calls), **kwargs)
        std_message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            images=images,
            response_info=response_info
        )        
        return std_message, client_message

    
    def extract_stream_chunks(self, response: GenerateContentResponse, **kwargs) -> Generator[MessageChunk, None, tuple[Message, Content]]:
        for chunk in response:
            # content = ""
            # tool_calls = []
            # images = []
            # for part in chunk.parts:
            #     if part.text:
            #         content += part.text
            #     elif part.function_call:
            #         tool_calls.append(self.extract_tool_call(part.function_call))
            #     elif part.inline_data:
            #         images.append(self.extract_image(part.inline_data))
            #     elif part.file_data or part.executable_code or part.code_execution_result:
            #         raise NotImplementedError("file_data, executable_code, and code_execution_result are not yet supported by UnifAI")
            content, tool_calls, images = self._extract_parts(chunk.parts)

            yield MessageChunk(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                images=images
            )
        
        return self.extract_assistant_message_both_formats(response, **kwargs)


    # def split_tool_outputs_into_messages(self, 
    #                                      tool_calls: list[ToolCall], 
    #                                      content: Optional[str] = None) -> Iterator[Message]:
    #     yield Message(
    #         role="tool",
    #         content=content,
    #         tool_calls=tool_calls       
    #     )


    # List Models
    def list_models(self) -> list[str]:
        return [model.name[7:] for model in self.client.list_models()]

    def format_model_name(self, model: str) -> str:
        if model.startswith("models/"):
            return model
        return f"models/{model}"

    # # Chat
    # def chat(
    #         self,
    #         messages: list[Message],     
    #         model: Optional[str] = None,
    #         system_prompt: Optional[str] = None,                   
    #         tools: Optional[list[Any]] = None,
    #         tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,            
    #         response_format: Optional[Union[str, dict[str, str]]] = None,

    #         max_tokens: Optional[int] = None,
    #         frequency_penalty: Optional[float] = None,
    #         presence_penalty: Optional[float] = None,
    #         seed: Optional[int] = None,
    #         stop_sequences: Optional[list[str]] = None, 
    #         temperature: Optional[float] = None,
    #         top_k: Optional[int] = None,
    #         top_p: Optional[float] = None, 

    #         **kwargs
    #         ) -> tuple[Message, Any]:
        
    #     model = self.format_model_name(model or self.default_model)
    #     gen_config = GenerationConfig(
    #         # candidate_count=1,
    #         stop_sequences=stop_sequences,
    #         max_output_tokens=max_tokens,
    #         temperature=temperature,
    #         top_k=top_k,
    #         top_p=top_p,
    #         response_mime_type=response_format, #text/plain or application/json
    #     )

    #     gen_model = self.client.GenerativeModel(
    #         model_name=model,
    #         safety_settings=kwargs.pop("safety_settings", None),
    #         generation_config=gen_config,
    #         tools=tools,
    #         tool_config=tool_choice,
    #         system_instruction=system_prompt,            
    #     )

    #     response = self.run_func_convert_exceptions(
    #         gen_model.generate_content,
    #         messages
    #     )
    #     return self.extract_assistant_message_both_formats(response, model=model)

    # Chat
    def get_chat_response(
            self,
            messages: list[Content],     
            model: Optional[str] = None,
            system_prompt: Optional[str] = None,                   
            tools: Optional[list[Any]] = None,
            tool_choice: Optional[Union[Tool, str, dict, Literal["auto", "required", "none"]]] = None,            
            response_format: Optional[Union[str, dict[str, str]]] = None,

            stream: bool = False,

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
        
        model = self.format_model_name(model or self.default_model)
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
        return self.run_func_convert_exceptions(
            gen_model.generate_content,
            messages,
            stream=stream,
            **kwargs
        )
        

    # Embeddings
    def embed(
            self,            
            input: str | Sequence[str],
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            **kwargs
            ) -> EmbedResult:
        
        if isinstance(input, str):
            input = [input]
        
        model=self.format_model_name(model or self.default_embedding_model)
        response = self.run_func_convert_exceptions(
            self.client.embed_content,
            content=input,
            model=model,
            output_dimensionality=max_dimensions,
            **kwargs
        )
        embeddings = [Embedding(vector=vector, index=i) for i, vector in enumerate(response["embedding"])]
        response_info = ResponseInfo(model=model, usage=Usage())
        return EmbedResult(embeddings=embeddings, response_info=response_info)
            
    
    


    def create_assistant(self, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")
        
    def update_assistant(self, ass_id, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def create_thread(self):
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def create_run(self):
        raise NotImplementedError("This method must be implemented by the subclass")    
    


   
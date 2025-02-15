from typing import Optional, Union, Sequence, Any, Literal, Mapping, Iterator, Generator

import google.generativeai as genai
from google.generativeai.types import (
    GenerateContentResponse,
    GenerationConfig,
    ContentType,
    BlockedPromptException,
    StopCandidateException,
    IncompleteIterationError,
    BrokenResponseError,
)   
    
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
    ToolConfig as GoogleToolConfig,
    Candidate,
    Content,
)
from google.protobuf.struct_pb2 import Struct as GoogleProtobufStruct

from ...types import (
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
)
from ..adapters.google_adapter import GoogleAdapter
from .._base_components._base_llm import LLM
from ...utils import generate_random_id
from ...exceptions import ProviderUnsupportedFeatureError

class GoogleLLM(GoogleAdapter, LLM):
    provider = "google"
    default_llm_model = "gemini-2.0-flash-exp"

    _system_prompt_input_type = "kwarg"
    
    # List Models
    def _list_models(self) -> list[str]:
        return GoogleAdapter._list_models(self)

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
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_mime_type=response_format, #text/plain or application/json
            # response_schema=response_format, # TODO use for json_schema
        )

        gen_model = self.client.GenerativeModel(
            model_name=model,
            safety_settings=kwargs.pop("safety_settings", None),
            generation_config=gen_config,
            tools=tools,
            tool_config=tool_choice,
            system_instruction=system_prompt,            
        )
        return gen_model.generate_content(
            messages,
            stream=stream,
            **kwargs
        )


    # Convert Objects from UnifAI to AI Provider format        
        # Messages    
    def format_user_message(self, message: Message) -> Any:
        parts = []
        if message.content:
            parts.append(Part(text=message.content))
        if message.images:
            parts.extend(map(self.format_image, message.images))
        return {"role": "user", "parts": parts}

    def format_assistant_message(self, message: Message) -> Any:
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

            schema_kwargs = {
                "type": tool_parameter.type.upper(),
                "description": tool_parameter.description,
                "nullable": nullable, 
            }
            if tool_parameter.enum:
                if not isinstance(tool_parameter, StringToolParameter):
                    # TODO maybe handle non-string enums by converting to string before and back after
                    raise ProviderUnsupportedFeatureError(f"Google LLM only supports enum when ToolParameter.type is string Got {tool_parameter.type} with enum: {tool_parameter.enum}")
                else:
                    schema_kwargs["enum"] = tool_parameter.enum
            if isinstance(tool_parameter, ObjectToolParameter):
                schema_kwargs["properties"] = {
                    param.name: tool_parameter_to_schema(param) 
                    for param in tool_parameter.values()
                }
                schema_kwargs["required"] = tool_parameter.required
            elif isinstance(tool_parameter, ArrayToolParameter):
                schema_kwargs["items"] = tool_parameter_to_schema(tool_parameter.items)
            return Schema(**schema_kwargs)
        
        return GoogleTool(function_declarations=[
            FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool_parameter_to_schema(tool.parameters) if tool.parameters else None,
            )]
        )
    
        # Tool Choice            
    def format_tool_choice(self, tool_choice: str) -> GoogleToolConfig:
        if tool_choice in ("auto", "required", "none"):
            mode = tool_choice.upper() if tool_choice != "required" else "ANY"
            allowed_function_names = []
        else:
            mode = "ANY"
            allowed_function_names = [tool_choice,]
        
        return GoogleToolConfig(
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
    def parse_tool_call(self, response_tool_call: dict, **kwargs) -> ToolCall:
            return ToolCall(
                id=response_tool_call['id'] or f'call_{generate_random_id(24)}',
                tool_name=response_tool_call['name'],
                arguments=response_tool_call['args']
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
                output_tokens=response_obj.usage_metadata.candidates_token_count,
                cached_content_tokens=response_obj.usage_metadata.cached_content_token_count
            )

    def parse_response_info(self, response: GenerateContentResponse, **kwargs) -> ResponseInfo:        
        return ResponseInfo(
            model=kwargs.get("model"), 
            provider=self.provider,
            done_reason=self.parse_done_reason(response.candidates[0], **kwargs), 
            usage=self.parse_usage(response)
        )

        # Assistant Messages (Content, Images, Tool Calls, Response Info)
    def _response_to_dict(self, response: GenerateContentResponse) -> dict:
        _result = response._result
        return type(_result).to_dict(_result, use_integers_for_enums=False) # type: ignore (proto.MessageMeta.to_dict is incorrectly typed as returning a Message instead of a dict)

    def _extract_parts(self, response: GenerateContentResponse, candidate: int = 0) -> tuple[str|None, list[ToolCall]|None, list[Image]|None]:           
        """Extracts content, tool calls, and images from a Google GenerateContentResponse.
        Converts response to dict to avoid issues with Pydantic's handling of protobuf objects.

        This conversion is necessary to allow Pydantic to validate the protobuf objects,
        for example, attempting to convert a FunctionCall.args to a dict and then calling
        creating a BaseModel instance from it:
        ```
        response: GenerateContentResponse = ...
        function_call: FunctionCall = response.candidates[0].content.parts[0].function_call
        function_call_args = dict(function_call.args) # Values are still protobuf objects
        ```
        The problem arises when trying to instantiate a Pydantic model from a dict, contiaining
        protobuf objects, which will fail with:
        ```
        my_model_from_args = MyModel(**function_call_args) # Fails
        my_model_from_args = MyModel.model_validate(function_call_args) # Fails
        ```
        `TypeError: Repeated.__init__() missing 1 required keyword-only argument: 'marshal'`

        This occurs because only the top level is a true python dict, while nested elements remain as
        protobuf objects, which Pydantic cannot handle properly if the original protobuf object 
        has been garbage collected.
        
        Args:
            response (GenerateContentResponse): The response from Google's GenerateContent API
            candidate (int, optional): The candidate response to parse. Defaults to 0.
        Returns:
            tuple[str|None, list[ToolCall]|None, list[Image]|None]: A tuple containing:
                - content (str|None): The extracted text content, or None if empty
                - tool_calls (list[ToolCall]|None): List of parsed tool calls, or None if none present
                - images (list[Image]|None): List of parsed images, or None if none present
        Raises:
            NotImplementedError: If response contains unsupported part types beyond text, 
                               function_call, or inline_data
        """        
        response_dict = self._response_to_dict(response)
        dict_parts = response_dict["candidates"][candidate]["content"]["parts"]

        content = None
        tool_calls = None
        images = None
        for part_dict in dict_parts:
            if part_text := part_dict.get("text"):
                if content is None:
                    content = ""
                content += part_text
            elif part_function_call := part_dict.get("function_call"):
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(self.parse_tool_call(part_function_call))
            elif part_inline_data := part_dict.get("inline_data"):
                if images is None:
                    images = []
                images.append(self.parse_image(part_inline_data))
            else:
                part_type = list(part_dict.keys())[0]
                raise NotImplementedError(
                    f"Google ProtoBuf Parts of type {part_type} are not yet supported by UnifAI. Only text, function_call, and inline_data are currently supported."
                )

        # Empty values (which can happen during streaming) should be returned as None
        return content, tool_calls, images

    def _is_non_empty_part(self, part: Part) -> bool:
        return bool(
            part.text 
            or part.function_call or part.function_response 
            or part.inline_data or part.file_data 
            or part.executable_code or part.code_execution_result
        )

    def parse_message(self, response: GenerateContentResponse, **kwargs) -> tuple[Message, Content]:
        client_message = response.candidates[0].content
        # Skip empty text contents at the end of ToolCall messages (role='model' with function_call parts) to prevent error:
        # "400 Unable to submit request because it has an empty text parameter. Add a value to the parameter and try again. 
        # Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini"
        client_message.parts = list(filter(self._is_non_empty_part, client_message.parts))
        
        # Extract content, tool calls, and images from the response AFTER converting to dict (see _extract_parts)
        content, tool_calls, images = self._extract_parts(response)

        response_info = self.parse_response_info(response, tools_called=bool(tool_calls), **kwargs)
        unifai_message = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            images=images,
            response_info=response_info
        )        
        return unifai_message, client_message
    
    def parse_stream(self, response: GenerateContentResponse, **kwargs) -> Generator[MessageChunk, None, tuple[Message, Content]]:
        parts = [] # To reassemble the final message after streaming
        for chunk in response:
            parts.extend(chunk.parts)
            content, tool_calls, images = self._extract_parts(chunk)
            
            if not content and not tool_calls and not images:
                continue

            yield MessageChunk(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                images=images
            )

        # Reset the original response parts back to collected parts (since they were consumed by the generator)
        # so it can be parsed as if it was a non-streaming response
        response.candidates[0].content.parts = parts
        return self.parse_message(response, **kwargs)
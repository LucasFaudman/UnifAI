import pytest
from basetest import base_test_llms, API_KEYS

from unifai import UnifAI, ProviderName
from unifai.types import Message, Tool, StringToolParameter, ToolCall
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

bad_param = StringToolParameter(
            type="string",
            required=True,
            description="This parameter is bad."
        )
bad_param.type = "bad_param"

bad_tool = Tool(
    name="bad_tool",
    type="function",
    description="This tool is bad.",
    parameters={
        "bad_param": bad_param
    }
)
bad_tool.type = "bad_tool"

bad_messages = [
    Message(role="assistant", tool_calls=[ToolCall(id="bad_id", tool_name="bad_tool", arguments={"bad_param": "bad_value"})])
]

@base_test_llms
@pytest.mark.parametrize("expected_exception, bad_init_kwargs, bad_chat_kwargs", [
    (APIConnectionError, {"base_url": "https://localhost:443/badapi"}, {}),
    (APITimeoutError, {"timeout": 0.0001}, {}),
    # (APIResponseValidationError, {}, {}),
    # (APIStatusError, {}, {}),
    (AuthenticationError, {"api_key": "bad_key"}, {}),
    (AuthenticationError, {"api_key": None}, {}),
    (AuthenticationError, {"api_key": b""}, {}),
    (BadRequestError, {}, {"tools": [bad_tool]}),
    (BadRequestError, {}, {"messages": bad_messages}),
    # (ConflictError, {}, {}),
    # (InternalServerError, {}, {}),
    (NotFoundError, {}, {"model": "bad_model"}),
    # (PermissionDeniedError, {}, {}),
    # (RateLimitError, {}, {"max_tokens": 1}),
    # (UnprocessableEntityError, {}, {}),
    # (UnknownAPIError, {}, {}),
])
def test_api_exceptions(
    provider: ProviderName, 
    init_kwargs: dict, 
    expected_exception: type[UnifAIError],
    bad_init_kwargs: dict,
    bad_chat_kwargs: dict,
    ):
    

    if provider == "ollama":
        if "base_url" in bad_init_kwargs:
            bad_init_kwargs["host"] = bad_init_kwargs.pop("base_url")

        if "api_key" in bad_init_kwargs:
            bad_init_kwargs["headers"] =  {"Authorization": f"Bearer {bad_init_kwargs.pop('api_key')}"} 
            return # Ollama doesn't have an API key to test
        
        if "tools" in bad_chat_kwargs:
            bad_chat_kwargs["model"] = "llama2-uncensored:latest" # Model exists but does not accept tools        
        
    if provider == "google":
        if "base_url" in bad_init_kwargs:
            expected_exception = UnifAIError
        if "timeout" in bad_init_kwargs:
            expected_exception = UnifAIError

    # init_kwargs.update(bad_init_kwargs)
    init_kwargs = {**init_kwargs, **bad_init_kwargs}

    
    bad_chat_kwargs["provider"] = provider
    bad_chat_kwargs["messages"] = [Message(role="user", content="What are all the exceptions you can return?")] 

    with pytest.raises(expected_exception):
        ai = UnifAI(api_keys=API_KEYS)
        messages = ai.chat(**bad_chat_kwargs)

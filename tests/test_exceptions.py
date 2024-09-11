import pytest
from basetest import base_test_all_providers

from unifai import UnifAIClient, AIProvider
from unifai.types import Message, FunctionTool, StringToolParameter
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

bad_tool = FunctionTool(
    name="bad_tool",
    type="function",
    description="This tool is bad.",
    parameters={
        "bad_param": bad_param
    }
)
bad_tool.type = "bad_tool"

bad_messages = [
    Message(role="system", content="Cant have"), 
    Message(role="system", content="two system messages")
]

@base_test_all_providers
@pytest.mark.parametrize("expected_exception, bad_client_kwargs, bad_func_kwargs", [
    (APIConnectionError, {"base_url": "https://localhost:443/badapi"}, {}),
    (APITimeoutError, {"timeout": 0.0001}, {}),
    # (APIResponseValidationError, {}, {}),
    # (APIStatusError, {}, {}),
    (AuthenticationError, {"api_key": "bad_key"}, {}),
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
    provider: AIProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    expected_exception: type[UnifAIError],
    bad_client_kwargs: dict,
    bad_func_kwargs: dict,
    ):
    

    if provider == "ollama":
        if "base_url" in bad_client_kwargs:
            bad_client_kwargs["host"] = bad_client_kwargs.pop("base_url")
        if "api_key" in bad_client_kwargs:
            bad_client_kwargs["headers"] =  {"Authorization": f"Bearer {bad_client_kwargs.pop('api_key')}"} 

    client_kwargs = {**client_kwargs, **bad_client_kwargs}
    
    func_kwargs["provider"] = provider
    func_kwargs["messages"] = [Message(role="user", content="What are all the exceptions you can return?")] 
    func_kwargs = {**func_kwargs, **bad_func_kwargs}

    print(f"provider:\n{provider}\n\nclient_kwargs:\n{client_kwargs}\n\nfunc_kwargs:\n{func_kwargs}")
    with pytest.raises(expected_exception):
        ai = UnifAIClient({provider: client_kwargs})
        ai.init_client(provider, **client_kwargs)
        messages = ai.chat(**func_kwargs)

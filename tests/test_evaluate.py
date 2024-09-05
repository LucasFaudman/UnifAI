import pytest
from unifai import UnifAIClient, AIProvider
from unifai._types import Message, Tool, EvalTypeParameters
from basetest import base_test_all_providers

TOOLS = {
    "return_flagged_and_reason": {
        "type": "function",
                "function": {
                    "name": "return_flagged_and_reason",
                    "description": "Return a boolean indicating whether the content should be flagged and a concise reason for the flag if True.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flagged": {
                                "type": "boolean",
                                "description": "True if the content should be flagged, False otherwise."
                            },
                            "reason": {
                                "type": "string",
                                "description": "A concise reason for the flag if True. An empty string if False."
                            }
                        },
                        "required": ["flagged", "reason"]
                    }
                }
    },
}

EVAL_TYPES = [
    EvalTypeParameters(
        name="urlEval",
        system_prompt=(
        "You review HTML text to flag elements that may contain spam, misinformation, or other malicious items. "
        "You also check the associated URLS for signs of typosquatting or spoofing. "
        # "Use the return_flagged_and_reason function to return your result."
        ),
        tools=["return_flagged_and_reason"],
        tool_choice="return_flagged_and_reason",
    )
]

@base_test_all_providers
@pytest.mark.parametrize("tools, tool_callables, eval_types, eval_type, content", [
    ([TOOLS["return_flagged_and_reason"]], None, EVAL_TYPES, "urlEval", {"url": "https://google.com", "link_text": "Google"}),
    ([TOOLS["return_flagged_and_reason"]], None, EVAL_TYPES, "urlEval", {"url": "https://g00gle.com", "link_text": "Google"}),
])
def test_evaluate_simple(
    provider: AIProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    tools: list,
    tool_callables: dict,
    eval_types: list,
    eval_type: str,
    content: str
    ):

    ai = UnifAIClient(
        provider_client_kwargs={provider: client_kwargs},
        tools=tools,
        tool_callables=tool_callables,
        eval_types=eval_types
    )
    ai.init_client(provider, **client_kwargs)

    response = ai.evaluate(
        eval_type=eval_type, 
        content=content,
        provider=provider,
        **func_kwargs
        )
    print(response)
    assert response

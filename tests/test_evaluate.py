import pytest
from unifai import UnifAIClient, LLMProvider
from unifai.types import Message, Tool
from unifai.client.specs import FuncSpec
from basetest import base_test_llms_all, PROVIDER_DEFAULTS

from pydantic import BaseModel

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


AI_FUNCS = [
    FuncSpec(
        name="urlEval",
        system_prompt=
        "You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. "
        "You check the associated URLS for signs of typosquatting or spoofing. ",
        # "Use the return_flagged_and_reason function to return your result.,
        tools=["return_flagged_and_reason"],
        tool_choice="return_flagged_and_reason",
        return_as="last_tool_call_args"
    )
]

@base_test_llms_all
@pytest.mark.parametrize("tools, tool_callables, func_specs, eval_name, content", [
    ([TOOLS["return_flagged_and_reason"]], None, AI_FUNCS, "urlEval", {"url": "https://google.com", "link_text": "Google"}),
    ([TOOLS["return_flagged_and_reason"]], None, AI_FUNCS, "urlEval", {"url": "https://g00gle.com", "link_text": "Google"}),
])
def test_evaluate_simple(
    provider: LLMProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    tools: list,
    tool_callables: dict,
    func_specs: list,
    eval_name: str,
    content: str
    ):

    ai = UnifAIClient(
        provider_client_kwargs={provider: client_kwargs},
        tools=tools,
        tool_callables=tool_callables,
        func_specs=func_specs
    )
    ai.init_client(provider, **client_kwargs)

    response = ai.evaluate(
        eval_spec=eval_name, 
        content=content,
        provider=provider,
        **func_kwargs
        )
    print(response)
    assert response


class FlaggedReason(BaseModel):
    flagged: bool
    reason: str

    def print_reason(self):
        print(f"Flagged: {self.flagged}\nReason: {self.reason}")

BASE_MODEL_AI_FUNCS = [
    FuncSpec(
        name="urlEval-BaseModel",
        system_prompt=
        "You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. "
        "You check the associated URLS for signs of typosquatting or spoofing. "
        "Use the return_flagged_and_reason function to return your result.",
        tools=["return_flagged_and_reason"],
        tool_choice="return_flagged_and_reason",
        return_on="message",
        return_as="last_tool_call_args",
        output_parser=FlaggedReason,
        prompt_template="URL:{url}\nLINK TEXT:{link_text}"
    ),

]



@base_test_llms_all
@pytest.mark.parametrize("tools, tool_callables, func_specs, eval_name, content, flagged", [
    ([TOOLS["return_flagged_and_reason"]], None, BASE_MODEL_AI_FUNCS, "urlEval", {"url": "https://google.com", "link_text": "Google"}, False),
    ([TOOLS["return_flagged_and_reason"]], None, BASE_MODEL_AI_FUNCS, "urlEval", {"url": "https://g00gle.com", "link_text": "Google"}, True),
])
def test_evalutate_base_model(
    provider: LLMProvider, 
    client_kwargs: dict, 
    func_kwargs: dict,
    tools: list,
    tool_callables: dict,
    func_specs: list,
    eval_name: str,
    content: dict,
    flagged: bool
    ):

    ai = UnifAIClient(
        provider_client_kwargs={
            provider: client_kwargs,
            "openai": PROVIDER_DEFAULTS["openai"][1]
            },
        tools=tools,
        tool_callables=tool_callables,
        func_specs=func_specs
    )
    url_eval = ai.get_function("urlEval-BaseModel")
    response = url_eval(url=content["url"], link_text=content["link_text"])
    assert response.flagged == flagged
    assert response.reason if flagged else not response.reason
    assert isinstance(response, FlaggedReason)
    print(f"{response.flagged=} {response.reason=}")
    response.print_reason()

import pytest
from unifai import UnifAI, FunctionConfig, BaseModel
from unifai.components.output_parsers.pydantic_output_parser import PydanticParser
from unifai.components.prompt_templates import PromptTemplate, PromptModel
from unifai.configs.rag_config import RAGConfig, RAGPrompterConfig, leave_query_as_is, default_rag_prompt_template, RAGPromptTemplate, RAGPromptModel
from unifai.types import Message, Tool, ArrayToolParameter, ObjectToolParameter, BooleanToolParameter, StringToolParameter, NumberToolParameter
from basetest import base_test_llms, API_KEYS
from unifai.types.annotations import ProviderName
from typing import Literal
import httpx


@pytest.mark.parametrize("url, link_text, flagged", [
    ("https://google.com", "Google", False),
    ("https://g00gle.com", "Google", True),
    # ("https://github.com", "GitHub", False),
    # ("https://githu8.com", "GitHub", True),
    # ("https://microsoft.com", "Microsoft", False),
    # ("https://micros0ft.com", "Microsoft", True),
    # ("https://apple.com", "Apple", False),
    # ("https://app1e.com", "Apple", True),    
    # ("chromeupdater.com", "Chrome Updater", True),
])
@base_test_llms
def test_evalutate_flagged_reason(
    provider: ProviderName, 
    init_kwargs: dict, 
    url, 
    link_text,
    flagged: bool
    ):

    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])

    class FlaggedReason(BaseModel):
        flagged: bool
        """True if the content should be flagged, False otherwise."""
        reason: str
        """A concise reason for the flag if True. An empty string if False."""

        def print_reason(self):
            print(f"Flagged: {self.flagged}\nReason: {self.reason}")

    url_eval_config_template = FunctionConfig(
        name="urlEval",
        system_prompt="You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. Check the associated URLS for signs of typosquatting or spoofing. ",
        input_parser=PromptTemplate("URL:{url}\nLINK TEXT:{link_text}"),
        output_parser=FlaggedReason,
    )

    url_eval = ai.function(url_eval_config_template)
    flagged_reason = url_eval(url=url, link_text=link_text)
    assert flagged_reason.flagged == flagged
    assert isinstance(flagged_reason.reason, str)
    assert isinstance(flagged_reason, FlaggedReason)
    print(f"{flagged_reason.flagged=} {flagged_reason.reason=}")
    flagged_reason.print_reason()


    class UrlEvalPrompt(PromptModel):
        "URL:{url}\nLINK TEXT:{link_text}"
        url: str
        link_text: str

    url_eval_config_prompt = FunctionConfig(
        name="urlEval",
        system_prompt="You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. Check the associated URLS for signs of typosquatting or spoofing. ",
        input_parser=UrlEvalPrompt,
        output_parser=FlaggedReason,
    )

    url_eval = ai.function(url_eval_config_prompt)
    flagged_reason = url_eval(url=url, link_text=link_text)
    assert flagged_reason.flagged == flagged
    assert isinstance(flagged_reason.reason, str)
    assert isinstance(flagged_reason, FlaggedReason)
    print(f"{flagged_reason.flagged=} {flagged_reason.reason=}")
    flagged_reason.print_reason()

    class UrlEvalPrompt(PromptModel):
        "URL:{url}\nLINK TEXT:{link_text}"
        url: str
        link_text: str    

    class UrlEvalResultPrompt(RAGPromptModel):
        """URL:{url}\nLINK TEXT:{link_text}\n\nCONTEXT:\n{result}"""
        url: str
        link_text: str

    def make_prompt(url: str, link_text: str) -> str:
        return f"URL:{url}\nLINK TEXT:{link_text}"

    prompter_config = RAGConfig(
        query_modifier=UrlEvalPrompt,
        # query_modifier=UrlEvalResultPrompt,
        # prompt_template=RAGPromptTemplate(template="URL:{url}\nLINK TEXT:{link_text}\n\nCONTEXT:\n{result}"),
        # query_modifier=make_prompt,
        prompt_template=UrlEvalResultPrompt,
        vector_db="chroma",        
        
    )
    prompter = ai.ragpipe(prompter_config)

    fn_with_rag = ai.function(FunctionConfig(
        input_parser=prompter_config,
        output_parser=FlaggedReason,
    ))
    flagged_reason = fn_with_rag(url=url, link_text=link_text)
    assert flagged_reason.flagged == flagged
    assert isinstance(flagged_reason.reason, str)
    assert isinstance(flagged_reason, FlaggedReason)
    print(f"{flagged_reason.flagged=} {flagged_reason.reason=}")
    flagged_reason.print_reason()    
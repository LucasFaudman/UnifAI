import pytest
from unifai import UnifAI, FunctionConfig, RAGConfig, BaseModel
from unifai.components.output_parsers.pydantic_output_parser import PydanticParser
from unifai.components.prompt_templates import PromptTemplate, PromptModel
from unifai.components.prompt_templates.rag_prompt_model import RAGPromptModel
from unifai.components.prompt_templates.rag_prompt_template import RAGPromptTemplate
from unifai.types import Message, Document
from basetest import base_test_llms, API_KEYS
from unifai.types.annotations import ProviderName
from typing import Literal

# Common test data
TEST_URLS = [
    ("https://google.com", "Google", False),
    ("https://g00gle.com", "Google", True),
]

# Common FlaggedReason model
class FlaggedReason(BaseModel):
    flagged: bool
    """True if the content should be flagged, False otherwise."""
    reason: str
    """A concise reason for the flag if True. An empty string if False."""

    def print_reason(self):
        print(f"Flagged: {self.flagged}\nReason: {self.reason}")

@pytest.mark.parametrize("url, link_text, flagged", TEST_URLS)
@base_test_llms
def test_basic_function_template(provider: ProviderName, init_kwargs: dict, url: str, link_text: str, flagged: bool):
    """Test basic function with PromptTemplate"""
    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    
    config = FunctionConfig(
        name="urlEval",
        system_prompt="You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. Check the associated URLS for signs of typosquatting or spoofing.",
        input_parser=PromptTemplate("URL:{url}\nLINK TEXT:{link_text}"),
        output_parser=FlaggedReason,
    )
    
    url_eval = ai.function(config)
    result = url_eval(url=url, link_text=link_text)
    
    assert result.flagged == flagged
    assert isinstance(result.reason, str)
    assert isinstance(result, FlaggedReason)

@pytest.mark.parametrize("url, link_text, flagged", TEST_URLS)
@base_test_llms
def test_prompt_model(provider: ProviderName, init_kwargs: dict, url: str, link_text: str, flagged: bool):
    """Test function with PromptModel"""
    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    
    class UrlEvalPrompt(PromptModel):
        "URL:{url}\nLINK TEXT:{link_text}"
        url: str
        link_text: str
    
    config = FunctionConfig(
        name="urlEval",
        system_prompt="You review URLs and HTML text to flag elements that may contain spam, misinformation, or other malicious items. Check the associated URLS for signs of typosquatting or spoofing.",
        input_parser=UrlEvalPrompt,
        output_parser=FlaggedReason,
    )
    
    url_eval = ai.function(config)
    result = url_eval(url=url, link_text=link_text)
    
    assert result.flagged == flagged
    assert isinstance(result.reason, str)
    assert isinstance(result, FlaggedReason)

@pytest.mark.parametrize("url, link_text, flagged", TEST_URLS)
@base_test_llms
def test_rag_with_prompt_template(provider: ProviderName, init_kwargs: dict, url: str, link_text: str, flagged: bool):
    """Test RAG with PromptTemplate"""
    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    
    def make_prompt(url: str, link_text: str) -> str:
        return f"URL:{url}\nLINK TEXT:{link_text}"
    
    prompter_config = RAGConfig(
        name="fn_PromptTemplate",
        query_modifier=make_prompt,
        prompt_template=RAGPromptTemplate("URL:{url}\nLINK TEXT:{link_text}\n\nCONTEXT:\n{result}"),
        vector_db="chroma",
    )
    
    prompter = ai.ragpipe(prompter_config)
    prompter.vector_db.delete_all_collections()
    list(prompter.iupsert_documents([
        Document(id="1", text=f"URL: {url} IS_SAFE: {not flagged}", 
                metadata={"url": url, "link_text": link_text})
    ]))
    
    fn_with_rag = ai.function(FunctionConfig(
        name="urlEvalWithRag",
        input_parser=prompter_config,
        output_parser=FlaggedReason,
    ))
    
    result = fn_with_rag(url=url, link_text=link_text)
    assert result.flagged == flagged
    assert isinstance(result.reason, str)
    assert isinstance(result, FlaggedReason)

@pytest.mark.parametrize("url, link_text, flagged", TEST_URLS)
@base_test_llms
def test_rag_with_prompt_models(provider: ProviderName, init_kwargs: dict, url: str, link_text: str, flagged: bool):
    """Test RAG with PromptModel and RAGPromptModel"""
    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
    
    class UrlEvalPrompt(PromptModel):
        "URL:{url}\nLINK TEXT:{link_text}"
        url: str
        link_text: str    

    class UrlEvalResultPrompt(RAGPromptModel):
        """URL:{url}\nLINK TEXT:{link_text}\n\nCONTEXT:\n{result}"""
        url: str
        link_text: str
    
    prompter_config = RAGConfig(
        name="PromptModel_RAGPromptModel",
        query_modifier=UrlEvalPrompt,
        prompt_template=UrlEvalResultPrompt,
        vector_db="chroma",
    )
    
    prompter = ai.ragpipe(prompter_config)
    prompter.vector_db.delete_all_collections()
    list(prompter.iupsert_documents([
        Document(id="1", text=f"URL: {url} IS_SAFE: {not flagged}", 
                metadata={"url": url, "link_text": link_text})
    ]))
    
    fn_with_rag = ai.function(FunctionConfig(
        name="urlEvalWithRag",
        input_parser=prompter_config,
        output_parser=FlaggedReason,
    ))
    
    result = fn_with_rag(url=url, link_text=link_text)
    assert result.flagged == flagged
    assert isinstance(result.reason, str)
    assert isinstance(result, FlaggedReason)

@pytest.mark.parametrize("url, link_text, flagged", TEST_URLS)
@base_test_llms
def test_poem_generation(provider: ProviderName, init_kwargs: dict, url: str, link_text: str, flagged: bool):
    """Test poem generation and editing"""
    ai = UnifAI(api_keys=API_KEYS, provider_configs=[{"provider": provider, "init_kwargs": init_kwargs}])
        
    # Test basic poem generation
    poem_config = FunctionConfig(
        name="return_poem",
        system_prompt="You are a poet. Write a poem based on the theme and style of the joke.",
        input_parser=lambda topic: str(topic)
    )
    return_poem = ai.function(poem_config)
    poem = return_poem(topic="Jews")
    assert isinstance(poem.content, str)
    
    class PoemModel(BaseModel):
        poem_name: str
        author_name: str
        verses: list[str]
        topics: list[str]
        """Topics discussed in the poem"""
    
        @property
        def verse_count(self):
            return len(self.verses)
        
        def poem_string(self):
            return f"{self.poem_name}\n" + "\n".join(self.verses)

    # Test poem editing with FunctionConfig as input_parser
    edit_poem_config = FunctionConfig(
        name="edit_poem",
        system_prompt="You are a poet. Edit the poem to improve its style, rhythm, and rhyme.",
        input_parser=poem_config,
        output_parser=PoemModel
    )
    edit_poem = ai.function(edit_poem_config)
    poem_object = edit_poem("Dogs")
    assert isinstance(poem_object.poem_name, str)
    assert isinstance(poem_object.verses, list)
    print(poem_object.poem_string())
    assert all(isinstance(verse, str) for verse in poem_object.verses)

    # Test poem editing with Function as input_parser
    edit_poem_config = FunctionConfig(
        name="edit_poem",
        system_prompt="You are a poet. Edit the poem to improve its style, rhythm, and rhyme.",
        input_parser=return_poem,
        output_parser=PoemModel
    )
    edit_poem = ai.function(edit_poem_config)
    edited_poem = edit_poem("How do I connect to tor?")
    assert isinstance(edited_poem.poem_name, str)
    assert isinstance(edited_poem.verses, list)
    assert all(isinstance(verse, str) for verse in edited_poem.verses)    


    # Test poem editing with FunctionConfig as output_parser
    edit_poem_config = FunctionConfig(
        name="edit_poem",
        system_prompt="You are a poet. Edit the poem to improve its style, rhythm, and rhyme.",
        input_parser=lambda poem: str(poem),
        output_parser=PoemModel
    )
    poem_config = FunctionConfig(
        name="return_poem",
        system_prompt="You are a poet. Write a poem based on the theme and style of the joke.",
        input_parser=lambda input: str(input),
        output_parser=edit_poem_config
    )
    return_edited_poem = ai.function(poem_config)
    edited_poem = return_edited_poem("How do I connect to tor?")
    print(edited_poem)
    assert isinstance(edited_poem.poem_name, str)
    assert isinstance(edited_poem.verses, list)
    assert all(isinstance(verse, str) for verse in edited_poem.verses)
    print(edited_poem)


    # Test poem editing with Function as output_parser
    edit_poem = ai.function(edit_poem_config)
    poem_config = FunctionConfig(
        name="return_poem",
        system_prompt="You are a poet. Write a poem based on the theme and style of the joke.",
        input_parser=lambda input: str(input),
        output_parser=edit_poem
    )
    return_edited_poem = ai.function(poem_config)
    edited_poem = return_edited_poem("How do I connect to tor?")
    assert isinstance(edited_poem.poem_name, str)
    assert isinstance(edited_poem.verses, list)
    assert all(isinstance(verse, str) for verse in edited_poem.verses)
    print(edited_poem)    


    return_poem_no_input = FunctionConfig(
        name="return_poem_no_input",
        system_prompt="Write a poem about: ",
    )
    return_poem = ai.function(return_poem_no_input)
    poem = return_poem(input="Dogs")
    print(poem)

    return_poem = return_poem \
        .set_input_parser(lambda topic: str(topic)) \
        .set_output_parser(PoemModel)
    # return_poem.set_output_parser(PoemModel)
    # return_poem.output_parser = PoemModel

    poem = return_poem(topic="Cats")
    print(poem.verses)



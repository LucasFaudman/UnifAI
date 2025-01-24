from unifai import (
    UnifAI, 
    tool, 
    BaseModel, 
    PromptModel,
    RAGPromptModel,
    Document, 
    ProviderConfig, 
    DocumentLoaderConfig, 
    DocumentChunkerConfig, 
    EmbedderConfig, 
    VectorDBCollectionConfig, 
    RerankerConfig, 
    TokenizerConfig,
    RAGConfig, 
    FunctionConfig
)
from unifai.utils import clean_text

# from _provider_defaults import API_KEYS 
from typing import Iterable, Optional, ClassVar
from security import safe_command
from shlex import quote as shlex_quote
from subprocess import run
from pathlib import Path
import json
from argparse import ArgumentParser
from dotenv import load_dotenv
from os import getenv
from platform import platform

SHELL = getenv("SHELL", "/bin/bash")
load_dotenv() # Load api_keys from .env file

SAFE_COMMAND_RESTRICTIONS = [
    "PREVENT_ARGUMENTS_TARGETING_SENSITIVE_FILES"
]
MANSPLAIN_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/mansplain/mansplain_wd")

# Create a UnifAI instance with a persistent Chroma config as the VectorDB
ai = UnifAI(
    provider_configs=[
        ProviderConfig(
        provider="chroma",
        init_kwargs={
            "persist_directory": str(MANSPLAIN_PATH),         
            "is_persistent": True
        })
    ],
)

# Subclass of DocumentLoader to load manpages
BinaryName = str
def load_manpages(binaries: Iterable[BinaryName]) -> Iterable[Document]:
    for binary in binaries:
        # Get manpage for binary with man command
        output = safe_command.run(run, 
                                  ['man', binary], 
                                  restrictions=SAFE_COMMAND_RESTRICTIONS,
                                  capture_output=True
                                )
        if not output or not output.stdout:
            print(f"Manpage for {binary} not found")
            continue
        # Yield a Document with the manpage id, text, and metadata
        yield Document(
            id=f"{binary}_manpage", 
            text=clean_text(output.stdout.decode("utf-8")), 
            metadata={"binary": binary}
        )
                                  

# Now the ManpageDocumentLoader can be accessed by name inside UnifAIComponentConfig(s)
rag_config = RAGConfig(
    name="manpage_rag",
    document_loader=DocumentLoaderConfig(load_documents=load_manpages),
    document_chunker=DocumentChunkerConfig(
        # chunk_size=1000,
        separators=["\n\n", "\n"], 
        keep_separator="start", 
    ),
    vector_db=VectorDBCollectionConfig(
        provider="chroma", 
        name="manpage_collection", 
        embedder="openai", 
        embedding_model="text-embedding-3-large"
    ),
    reranker=RerankerConfig(
        provider="rank_bm25", 
        default_model="BM25Okapi"
    ),
    tokenizer=TokenizerConfig(
        provider="tiktoken", 
        default_model="gpt2"
    ),
)

# Create a RAGPipe from the RAGConfig
ragpipe = ai.ragpipe(rag_config)

def get_already_ingested_binaries() -> set[BinaryName]:
    return set(id.split('_')[0] for id in ragpipe.vector_db_collection.list_ids())
    
class ManspainSystemPrompt(PromptModel):
    """
    You are a command line expert who is a prolific mansplainer. 
    Your role is to use your expertise and relevent manpages to assist users with command line tasks,
    fix broken commands, and answer questions with detailed explanations and command suggestions.
    ALWAYS include relevant details from the manpages in your responses.

    Your responses should be tailored to the following platform and shell:
    Platform: {platform}
    Shell: {shell}

    Output Formatting Instructions for your response content (NOT TOOL CALLS):    
    Format your responses to be viewed in terminal easily with terminal escape sequences for color and formatting. 
    All binary should be bolded and given a unique color. Example: \033[1;32mbinary1\033[0m \033[1;34mbinary2\033[0m 
    Use the following colors in order for each binary: 32, 34, 36, 31, 33, 35.
    Only reuse colors if you run out of unique colors and begin again with the first color.
    Section Headers should ALWAYS be underlined and bolded. Example: \033[1;4mheader\033[0m 
    IMPORTANT: NEVER USE # header, ### header or ANY markdown formatting for any reason.
    Bullet point titles should ALWAYS be bolded and colored consistently if the same title is resused. Example: \033[1;32mtitle\033[0m 
    IMPORTANT: NEVER USE **title**, *title* or ANY markdown formatting for any reason.
    Code snippets / Commands should ALWAYS be bolded and white. Example: \033[1;37mcommand\033[0m
    CRITICAL: NEVER USE ANY markdown formatting including *TEXT*, **TEXT**, ### TEXT, `CODE`, ```shell CODE``` or similar. ONLY USE terminal escape formatting since your responses will be viewed in a terminal.
    CRITICAL: DO NOT INCLUDE ANY FORMATTING IN TOOL CALLS. ONLY USE PLAIN TEXT IN TOOL CALLS.
    
    {tone}
    """
    platform: str = platform()
    shell: str = SHELL
    tone: Optional[str] = None
    value_formatters = {
        "tone": lambda tone: f"\nYour tone should be: {tone}" if tone else ""

    }

class ManpageRagPrompt(RAGPromptModel):
    """Mansplain the user query using relevant manpages
    {query}
    {result}
    """
    value_formatters = {
        "query": lambda query: f"User query: {query}",
        "result": lambda result: "Relevant Manpages:\n" + "\n".join(f"{doc.id}\n{doc.text}" for doc in result) 
    }

mansplain_query = ai.function(FunctionConfig(
    name="mansplain_query",
    system_prompt=ManspainSystemPrompt,
    input_parser=ragpipe.with_prompt_template(ManpageRagPrompt),
))



class SuggestBinariesPrompt(PromptModel):
    """
    Suggest binaries that can be used to complete the task described in the query.
    User query: 
    {query}
    """
    query: str

class BinarySuggestion(BaseModel):
    binary: str
    """The binary to run. ie grep, ls, curl, nmap, etc"""
    description: str
    """A description of the binary and its purpose"""
    reasoning_for_suggestion: str
    """Your reasoning for suggesting this binary"""

    def __str__(self):
        return f"{self.binary}: {self.description}\n{self.reasoning_for_suggestion}"

class BinarySuggestions(BaseModel):
    """A list of binary suggestions in order of relevance to the user query"""
    suggestions: list[BinarySuggestion]
    pros_and_cons: str
    """A description of the pros and cons of the suggestions"""

    def __str__(self):
        return "Suggestions:\n" + "\n".join(f"\n{suggestion}" for suggestion in self.suggestions) + f"\nPros and Cons:\n{self.pros_and_cons}"

get_binary_suggestions = ai.function(FunctionConfig(
    name="get_binary_suggestions",
    system_prompt=ManspainSystemPrompt,
    input_parser=SuggestBinariesPrompt,
    output_parser=BinarySuggestions,
))



class CommandSuggestionPrompt(ManpageRagPrompt):
    """
    Suggest a command to complete the task described in the query
    {query}
    {result}
    """

class CommandSuggestion(BaseModel):
    """A command suggestion using a binary and arguments to complete the task or part of a command pipeline"""
    binary: str
    """The binary to run. ie grep, ls, curl, etc"""
    arguments: list[str]
    """A list of arguments to run the binary with"""
    description: str
    """A description of the command and its purpose"""
    reasoning_for_suggestion: str
    """Your reasoning for suggesting this command"""

    def command_string(self):
        return f"{self.binary} {' '.join(shlex_quote(arg) for arg in self.arguments)}"

    def __str__(self):
        return f"Command:\n{self.command_string()}\nDescription:\n{self.description}\nReasoning:\n{self.reasoning_for_suggestion}"



class CommandPipeline(BaseModel):
    """A full command string using one or more binaries and bash operators to complete the task. ie nc -l -p 1234 | grep "search term" """    
    explanation: str
    """A detailed explanation of the command pipeline and its purpose including the use of each binary all operators | > < && || etc"""
    commands: list[CommandSuggestion]
    full_command_string_to_run: str
    """The full command string Literal to run in the terminal. Should not be quoted or escaped as it will be the literal command string to run."""

    def __str__(self):
        return f"{self.explanation}\nCommands:\n" + "\n".join(f"\n{command}" for command in self.commands) + f"\nFull Command String:\n{self.full_command_string_to_run}"

get_command_suggestion = ai.function(FunctionConfig(
    name="get_command_suggestion",
    system_prompt=ManspainSystemPrompt,
    input_parser=ragpipe.with_prompt_template(CommandSuggestionPrompt),
    output_parser=CommandPipeline,
))

class CommandFixPrompt(RAGPromptModel):
    """Fix a broken command in the query command string.
    {query}    
    Original command: 
    {command_to_fix}
    Error:
    {error}
    Relevant Manpages:
    {result}
    """
    command_to_fix: str
    error: Optional[str] = None
        
class FixedCommand(BaseModel):
    fix_explanation: str
    """A detailed explanation of the error and how it was fixed"""
    original_command: str
    """The original command string"""
    fixed_command: CommandPipeline

    def __str__(self):
        return f"{self.fix_explanation}\nOriginal Command:\n{self.original_command}\nFixed Command:\n{self.fixed_command}"

fix_command = ai.function(FunctionConfig(
    name="fix_command",
    system_prompt=ManspainSystemPrompt,
    input_parser=ragpipe.with_prompt_template(CommandFixPrompt),
    output_parser=FixedCommand,
))




if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Mansplain command line tasks and answer questions with detailed explanations and command suggestions")
    # positional argument for query 
    arg_parser.add_argument("query", type=str, nargs='?', help="The user query to mansplain")
    arg_parser.add_argument("-b", "--binaries", type=str, nargs="*", help="The binaries who's manpages to use to mansplain the query")
    arg_parser.add_argument("-sb", "--suggest-binaries", action="store_true", default=False, help="Get suggestions for binaries to use")
    arg_parser.add_argument("-sc", "--suggest-command", action="store_true", default=False, help="Get command suggestions")
    arg_parser.add_argument("-u", "--ingest", action="store_true", default=True, help="Ingest manpages for the provided binaries")
    arg_parser.add_argument("-f", "--fix", type=str, nargs='?', const=True, help="A broken command to fix")
    arg_parser.add_argument("-e", "--error", type=str, nargs='?', const=True, help="Error message from the broken command")
    arg_parser.add_argument("-r", "--run", action="store_true", help="Run the suggested or fixed command")
    arg_parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm running the command")
    
    arg_parser.add_argument_group("Mansplain System Prompt Arguments")
    arg_parser.add_argument("--tone", type=str, help="The tone of the mansplain")

    ingest_group = arg_parser.add_argument_group("Ingestion Arguments")
    ingest_group.add_argument("--chunk-size", type=int, default=1000, help="Size of chunks for splitting manpages")
    ingest_group.add_argument("--chunk-overlap", type=float, default=0.2, help="Overlap between chunks as a fraction of chunk size")

    rag_group = arg_parser.add_argument_group("RAG Arguments")
    rag_group.add_argument("--top-k", type=int, default=20, help="Number of documents to retrieve from vector store")
    rag_group.add_argument("--top-n", type=int, default=5, help="Number of documents to return after reranking")
        
    
    args = arg_parser.parse_args()

    query = args.query
    binaries = args.binaries or []
    suggest_binaries = args.suggest_binaries
    suggest_command = args.suggest_command

    if args.tone:
        system_prompt_kwargs = {"tone": args.tone}
        mansplain_query.system_prompt_kwargs = system_prompt_kwargs
        get_binary_suggestions.system_prompt_kwargs = system_prompt_kwargs
        get_command_suggestion.system_prompt_kwargs = system_prompt_kwargs
        fix_command.system_prompt_kwargs = system_prompt_kwargs    

    if suggest_binaries:
        if not query:
            print("No query provided to suggest binaries for. Please provide a query to suggest binaries for")
            exit()

        print("\nGetting binary suggestions...")
        binary_suggestions = get_binary_suggestions(query=query)
        print(f"\n{binary_suggestions}")
        binaries.extend(suggestion.binary for suggestion in binary_suggestions.suggestions)
    
    if args.ingest:
        binaries_to_ingest = set(binaries) - get_already_ingested_binaries()
        if binaries_to_ingest:
            chunk_kwargs = {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap}
            print("\nIngesting manpages for binaries: " + ', '.join(binaries_to_ingest))
            print(f"Chunk Size: {args.chunk_size} Chunk Overlap: {args.chunk_overlap}")
            i = -1
            for i, ingested_chunk in enumerate(ragpipe.ingest(binaries=binaries_to_ingest, chunk_kwargs=chunk_kwargs)):
                print(f"Ingested chunk {i}: {ingested_chunk.id} length: {len(ingested_chunk)} tokens: {ragpipe.tokenizer.count_tokens(text=ingested_chunk.text)}")        
            print(f"Done ingesting {i+1} manpage chunks for binaries: {', '.join(binaries_to_ingest)}")
        else:
            print(f"Manpages for binaries: {', '.join(binaries)} already ingested")

    if query:
        command = None
        where = {"binary": {"$in": binaries}} if binaries else None
        if suggest_command:
            print("\nGetting command suggestions...")
            command = get_command_suggestion(
                query=query, 
                top_k=args.top_k, 
                top_n=args.top_n, 
                where=where
            )
            print(f"Suggested Command: {command}")
        elif (command_to_fix := args.fix):
            print(f"Fixing command: {command_to_fix}...")
            fixed_command = fix_command(
                command_to_fix=command_to_fix, 
                error=args.error, 
                top_k=args.top_k, 
                top_n=args.top_n, 
                where=where
            )
            print(f"Fixed Command: {fixed_command}")
            command = fixed_command.fixed_command
        else:
            print(f"\nMansplaining query: {query}")
            for message_chunk in mansplain_query.stream(
                query=query, 
                top_k=args.top_k, 
                top_n=args.top_n, 
                where=where):
                print(message_chunk.content, end="")
            print('Done mansplaining query')

    if args.run and command:
        print(f"\nCommand to run: {command.full_command_string_to_run}")
        if args.yes or input("Run the suggested command? (y/n): ").lower() == 'y':
            print("Running command...\n<BEGIN OUTPUT>")
            safe_command.run(run, 
                             command=command.full_command_string_to_run, 
                             restrictions=SAFE_COMMAND_RESTRICTIONS, 
                             shell=True
                        )
            print("<END OUTPUT>")
        else:
            print("Command not run")
    print("Mansplain complete")
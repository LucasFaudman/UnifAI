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

from typing import Iterable, Optional, ClassVar, NamedTuple
from security import safe_command
from shlex import quote as shlex_quote
from subprocess import run, Popen, PIPE
import sys
from pathlib import Path
import json
from textwrap import dedent
from argparse import ArgumentParser
from dotenv import load_dotenv
from os import getenv
from platform import platform
from threading import Thread
from queue import Queue

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

class ManpageRagPrompt(RAGPromptModel):
    """Mansplain the user query using relevant manpages
    {query}
    {result}
    """
    value_formatters = {
        "query": lambda query: f"User query: {query}",
        "result": lambda result: "Relevant Manpages:\n" + "\n".join(f"{doc.id}\n{doc.text}" for doc in result) 
    }                                  

# Configure the RAGPipe
rag_config = RAGConfig(
    name="manpage_rag",
    document_loader=DocumentLoaderConfig(load_documents=load_manpages),
    document_chunker=DocumentChunkerConfig(
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
    prompt_template=ManpageRagPrompt
)

# Create a RAGPipe from the RAGConfig
ragpipe = ai.ragpipe(rag_config)

def get_already_ingested_binaries() -> set[BinaryName]:
    return set(id.split('_')[0] for id in ragpipe.vector_db_collection.list_ids())
    
class ManspainSystemPrompt(PromptModel):
    """
    You are a command line expert and mansplainer. 
    Your role is to use your expertise and relevent manpages to assist users with command line tasks,
    fix broken commands, and answer questions with detailed explanations and command suggestions.
    ALWAYS include relevant details from the manpages in your responses.

    Your responses should be tailored to the following platform and shell:
    Platform: {platform}
    Shell: {shell}
    {formatting_instructions}
    {tone}
    """
    platform: str = platform()
    shell: str = SHELL
    formatting_instructions: Optional[str] = """
    Output Formatting Instructions for your response content (NOT TOOL CALLS):    
    Format your responses to be viewed in terminal easily with terminal escape sequences for color and formatting. 
    All binary should be bolded and given a unique color. Example: \033[1;32mbinary1\033[0m \033[1;34mbinary2\033[0m 
    Use the following colors in order for each binary: 32, 34, 36, 31, 33, 35.
    Only reuse colors if you run out of unique colors and begin again with the first color.
    Section Headers should ALWAYS be underlined and bolded. Example: \033[1;4mheader\033[0m 
    IMPORTANT: NEVER USE # header, ### header nor ANY markdown formatting in your response.
    Bullet point titles should ALWAYS be bolded and colored consistently if the same title is resused. Example: \033[1;32mtitle\033[0m 
    IMPORTANT: NEVER USE **title**, *title* nor ANY markdown formatting in your response.
    Code snippets / Commands should ALWAYS be bolded and white. Example: \033[1;37mcommand\033[0m
    CRITICAL: NEVER USE ANY markdown formatting including *TEXT*, **TEXT**, ### TEXT, `CODE`, ```shell CODE``` or similar. ONLY USE terminal escape formatting since your responses will be viewed in a terminal.
    CRITICAL: DO NOT INCLUDE ANY FORMATTING IN TOOL CALLS. ONLY USE PLAIN TEXT IN TOOL CALLS.
    """
    tone: Optional[str] = None
    value_formatters = {
        "formatting_instructions": lambda instructions: f"\n{instructions}" if instructions else "",
        "tone": lambda tone: f"\nYour tone should be: {tone}" if tone else "",
    }


class BinarySuggestionsPrompt(PromptModel):
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
    system_prompt=ManspainSystemPrompt(formatting_instructions=None),
    input_parser=BinarySuggestionsPrompt,
    output_parser=BinarySuggestions,
))


class CommandSuggestionPrompt(PromptModel):
    """
    Suggest a command to complete the task described in the query
    {query}
    """
    query: str

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
        return f"Command:\n\033[1;37m{self.command_string()}\033[0m\nDescription:\n{self.description}\nReasoning:\n{self.reasoning_for_suggestion}"


class CommandPipeline(BaseModel):
    """A full command string using one or more binaries and bash operators to complete the task. ie nc -l -p 1234 | grep "search term" """    
    explanation: str
    """A detailed explanation of the command pipeline and its purpose including the use of each binary all operators | > < && || etc"""
    commands: list[CommandSuggestion]
    full_command_string_to_run: str
    """The full command string Literal to run in the terminal. Should not be quoted or escaped as it will be the literal command string to run."""

    def __str__(self):
        return f"{self.explanation}\nCommands:\n" + "\n".join(f"\n{command}" for command in self.commands) + f"\nFull Command String:\n{self.full_command_string_to_run}"


class CommandFixPrompt(PromptModel):
    """Fix a broken command.
    Original command: 
    {command_to_fix}
    {output}
    """
    command_to_fix: str
    output: Optional[str] = None
    value_formatters = {
        "output": lambda output: f"Output:\n{output}" if output else ""
    }
        
class FixedCommand(BaseModel):
    fix_explanation: str
    """A detailed explanation of the error and how it was fixed"""
    original_command: str
    """The original command string"""
    fixed_command: CommandPipeline

    def __str__(self):
        return f"{self.fix_explanation}\nOriginal Command:\n{self.original_command}\nFixed Command:\n{self.fixed_command}"


mansplain_chat_function = ai.function(FunctionConfig(
    name="mansplain_chat_function",
    stateless=False,
    system_prompt=ManspainSystemPrompt,
    input_parser=ragpipe
))


class CommandResult(BaseModel):
    output: str
    stdout: str
    stderr: str
    return_code: int

def run_command_with_output(command: str, capture: bool = False) -> CommandResult:
    """Run a command and handle output in real-time while optionally capturing it.
    
    Args:
        command: The command string to run
        capture: Whether to capture the output for later use
        
    Returns:
        CommandResult containing stdout, stderr and return code
    """
    # Create queues for stdout and stderr
    output_queue = Queue()
    combined_output = []
    captured_stdout = []
    captured_stderr = []
    
    def handle_stream(stream, stream_name):
        for line in iter(stream.readline, b''):
            line = line.decode().rstrip()
            output_queue.put((stream_name, line))
            if capture:
                if stream_name == 'stdout':
                    captured_stdout.append(line)
                else:
                    captured_stderr.append(line)
        stream.close()

    # Start the process with safe_command wrapper
    process = safe_command.run(
        Popen,
        command,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        bufsize=1,
        restrictions=SAFE_COMMAND_RESTRICTIONS
    )
    assert process is not None # "safe_command.run returned None"

    # Create and start stream handler threads
    stdout_thread = Thread(target=handle_stream, args=(process.stdout, 'stdout'))
    stderr_thread = Thread(target=handle_stream, args=(process.stderr, 'stderr'))
    stdout_thread.start()
    stderr_thread.start()
    
    print("<BEGIN OUTPUT>")
    # Process output as it comes in
    while process.poll() is None or not output_queue.empty():
        try:
            stream_name, line = output_queue.get(timeout=0.1)
            combined_output.append(line)
            if stream_name == 'stdout':
                print(line)
            else:
                print(line, file=sys.stderr)
        except:
            continue
            
    # Wait for threads and process to complete
    stdout_thread.join()
    stderr_thread.join()
    return_code = process.wait()
    print("<END OUTPUT>")
    
    # Return captured output if requested
    return CommandResult(
            output='\n'.join(combined_output),
            stdout='\n'.join(captured_stdout),
            stderr='\n'.join(captured_stderr),
            return_code=return_code
    )


if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Mansplain command line tasks and answer questions with detailed explanations and command suggestions")
    arg_parser.add_argument("query", type=str, nargs='?', help="The user query to mansplain")
    arg_parser.add_argument("-b", "--binaries", type=str, nargs="*", help="The binaries who's manpages to use to mansplain the query")
    arg_parser.add_argument("-sb", "--suggest-binaries", action="store_true", default=False, help="Get suggestions for binaries to use")
    arg_parser.add_argument("-sc", "--suggest-command", action="store_true", default=False, help="Get command suggestions")
    arg_parser.add_argument("-u", "--ingest", action="store_true", default=True, help="Ingest manpages for the provided binaries")
    arg_parser.add_argument("-f", "--fix", type=str, nargs='?', const=True, help="A broken command to fix")
    arg_parser.add_argument("-e", "--error", type=str, nargs='?', const=True, help="Error message from the broken command")
    arg_parser.add_argument("-r", "--run", action="store_true", help="Run the suggested or fixed command")
    arg_parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm running the command")
    arg_parser.add_argument("-c", "--capture", action="store_true", help="Capture command output to use in subsequent commands")
    arg_parser.add_argument("-rp", "--replace", nargs="+", help="Replacement pairs in format: pattern1 replacement1 pattern2 replacement2 ...")
    
    arg_parser.add_argument_group("Mansplain System Prompt Arguments")
    arg_parser.add_argument("-t", "--tone", type=str, help="The tone of the mansplain")

    ingest_group = arg_parser.add_argument_group("Ingestion Arguments")
    ingest_group.add_argument("--chunk-size", type=int, default=1000, help="Size of chunks for splitting manpages")
    ingest_group.add_argument("--chunk-overlap", type=float, default=0.2, help="Overlap between chunks as a fraction of chunk size")

    rag_group = arg_parser.add_argument_group("RAG Arguments")
    rag_group.add_argument("-n", "--top-k", type=int, default=30, help="Number of documents to retrieve from vector store")
    rag_group.add_argument("-k", "--top-n", type=int, default=10, help="Number of documents to return after reranking")
        
    
    args = arg_parser.parse_args()

    query = args.query
    binaries = args.binaries or []
    suggest_binaries = args.suggest_binaries
    replacements = {}
    if args.replace:
        replacements = dict(zip(args.replace[::2], args.replace[1::2]))
        print(f"Replacements: {replacements}")
        
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

    suggest_command = args.suggest_command
    command_to_fix = args.fix
    command = None
    error = args.error
    while query or suggest_command or command_to_fix:
        mansplain_chat_function = mansplain_chat_function.set_parsers(ragpipe.prompt)
        mansplain_chat_function.tools = mansplain_chat_function.tool_choice = None
        mansplain_chat_function.system_prompt_kwargs = {"tone": args.tone}
        
        if query:
            print(f"\nMansplaining query: {query} using manpages for binaries: {', '.join(binaries)}...")
            out_message = mansplain_chat_function.print_stream(
                query=query, 
                top_k=args.top_k, 
                top_n=args.top_n, 
                where={"binary": {"$in": binaries}} if binaries else None,
                replacements=replacements
            )
            print(f'\n\nDone mansplaining.\nResponse Info:\n{out_message.response_info}')

            if suggest_command:
                print("\nGetting command suggestions...")
                get_command_suggestion = mansplain_chat_function.set_parsers(CommandSuggestionPrompt, CommandPipeline)
                get_command_suggestion.system_prompt_kwargs = {"formatting_instructions": None}
                command = get_command_suggestion.print_stream(query=query, replacements=replacements)
                print(f"Suggested Command: {command}")

        if command_to_fix and not suggest_command:
            print(f"Fixing command: {command_to_fix}...")
            fix_command = mansplain_chat_function.set_parsers(CommandFixPrompt, FixedCommand)
            fix_command.system_prompt_kwargs = {"formatting_instructions": None}
            fixed_command = fix_command.print_stream(command_to_fix=command_to_fix, output=error, replacements=replacements)
            print(f"Fixed Command: {fixed_command}")
            command = fixed_command.fixed_command

        if args.run and command:
            print(f"\nCommand to run:\n\033[1;37m{command.full_command_string_to_run}\033[0m")
            if args.yes or input("Run the suggested command? (y/n): ").lower() == 'y':
                result = run_command_with_output(command.full_command_string_to_run, args.capture)
                print(f"Command return code: {result.return_code}")
                if args.capture:
                    if input("Fix the command? (y/n): ").lower() == 'y':
                        command_to_fix = command.full_command_string_to_run
                        error = result.output
                        query = input(f"Original query: {query}\nEnter new query or press ENTER to keep original: ")
                        suggest_command = False
                        continue
            else:
                print("Command not run")
        
        if (query := input(f"Original query: {query}\nEnter new query or press ENTER to keep original: ")):            
            suggest_command = input("Suggest a command for the query? (y/n): ").lower() == 'y'
            command_to_fix = False
            if input("Reset the chat? (y/n): ").lower() == 'y':
                mansplain_chat_function.reset()


    print("Mansplain complete")
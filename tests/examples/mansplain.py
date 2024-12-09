from unifai import UnifAI, tool, BaseModel, ProviderConfig, DocumentChunkerConfig, VectorDBCollectionConfig, EmbedderConfig, RAGConfig, Document, DocumentLoaderConfig, RerankerConfig, FunctionConfig
from unifai.components._base_components._base_document_loader import DocumentLoader

from _provider_defaults import API_KEYS 
from security import safe_command
from subprocess import run
from pathlib import Path
import json
MANSPLAIN_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/mansplain")

# Create a UnifAI instance with a persistent Chroma config as the VectorDB
ai = UnifAI(
    api_keys=API_KEYS, 
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
Manpage = str
class ManpageDocumentLoader(DocumentLoader[BinaryName, Manpage]):
    provider = "manpage_loader"

    def _load_source(self, source: BinaryName, *args, **kwargs) -> Manpage:
        output = safe_command.run(run, ['man', source], capture_output=True)
        if not output or not output.stdout:
            raise ValueError(f"Manpage for {source} not found")
        return output.stdout.decode(self.config.encoding)
    
    def _load_metadata(self, source: BinaryName, loaded_source: Manpage, metadata: dict|None, *args, **kwargs) -> dict|None:
        if metadata is None:
            metadata = {}
        metadata["binary_name"] = source
        return metadata
    
    def _process_id(self, source: BinaryName, loaded_source: Manpage, loaded_metadata: dict|None, *args, **kwargs) -> str:
        return f"{source}_manpage"

# Register the ManpageDocumentLoader with the UnifAI instance
ai.register_component(ManpageDocumentLoader)

# Now the ManpageDocumentLoader can be accessed by name inside UnifAIComponentConfig(s)
rag_config = RAGConfig(
    name="manpage_rag",
    document_loader="manpage_loader",
    document_chunker=DocumentChunkerConfig(separators=["\n\n", "\n"], keep_separator="start", chunk_size=1000),
    vector_db=VectorDBCollectionConfig(provider="chroma", name="manpage_collection", embedder="openai"),
    reranker=RerankerConfig(provider="rank_bm25", tokenizer="tiktoken"),
)

# Create a RAGPipe with the RAGConfig
ragpipe = ai.get_ragpipe(rag_config)

# Ingest the manpages for tor, curl, and nc. (Since the loader is the ManpageDocumentLoader, the source is the binary name)
for i, ingested_chunk in enumerate(ragpipe.ingest("tor", "curl", "nc")):
    print(f"Ingested chunk {i}: {ingested_chunk.id}")

# Define output models 
class CommandArgument(BaseModel):
    name: str
    """The argument name ie --help, install, -A, etc"""
    value: str
    """The argument value"""
    description: str
    """A description of the argument, its value, and its purpose"""

class CommandSuggestion(BaseModel):
    binary: str
    """The binary to run. ie grep, ls, curl, etc"""
    arguments: list[CommandArgument]
    description: str
    """A description of the command and its purpose"""

    def __str__(self) -> str:
        return f"{self.binary} '{' '.join([f'{arg.name} {arg.value}' for arg in self.arguments])}'"
    
class CommandSuggestions(BaseModel):
    suggestions: list[CommandSuggestion]
    pros_and_cons: str
    """A description of the pros and cons of the suggestions"""

    def __str__(self) -> str:
        return "\n".join([f"Suggestion {i}: {suggestion.description}\n{str(suggestion)}" for i, suggestion in enumerate(self.suggestions, start=1)])

# Define a function to get command suggestions using the RAGPipe
get_command_suggestions = ai.get_function(FunctionConfig(
    name="get_command_suggestions",
    system_prompt="You are a command line expert. Your role is to provide command suggestions and explaination base on the user's query and relevant context from manpages.",
    output_parser=CommandSuggestions,
    llm="openai",
    rag_config="manpage_rag", # Use the manpage_rag config from above
))

# Call the function to get a CommandSuggestions object
suggestions = get_command_suggestions("How do I connect to tor?")
print(suggestions)
print(suggestions.pros_and_cons)

suggestion0 = suggestions.suggestions[0]
args = suggestion0.arguments
command = suggestion0.binary
print(f"Command: {command} {' '.join([f'{arg.name} {arg.value}' for arg in args])}")



# LONG FORM Of Ingestion and Retrieval shown above

# Set up the components
# loader = ai.get_document_loader("manpage_loader")
# chunker = ai.get_document_chunker(DocumentChunkerConfig(separators=["\n\n", "\n"], keep_separator="start", chunk_size=1000))
# embedder = ai.get_embedder("openai")
# vector_db = ai.get_vector_db("chroma")
# manpage_collection = vector_db.get_or_create_collection(
#     name="manpage_collection",
#     distance_metric="cosine",
#     embedder="openai",
# )
# reranker = ai.get_reranker("rank_bm25", tokenizer="tiktoken")

# # Ingest manpages for tor, curl, and nc
# for binary_name in ["tor", "curl", "nc"]:
#     manpage = loader.load_document(binary_name)
#     chunks = chunker.chunk_document(manpage)
#     embedded_chunks = embedder.embed_documents(chunks)
#     manpage_collection.upsert_documents(embedded_chunks)

# # Query the vector db and rerank the results
# query = "How do I connect to tor?"
# query_embedding = embedder.embed(query)[0]
# query_result = manpage_collection.query(query_embedding, top_k=20)
# reranked_query_result = reranker.rerank(query, query_result, top_n=10)

# # Construct the prompt from the query and the reranked query result
# prompt = f"{query}\n\nCONTEXT:"
# for reranked_doc in reranked_query_result:
#     prompt += f"\nDOCUMENT: {reranked_doc.id}\n{reranked_doc.text}"

# print(prompt)


# Add back to above for loop to see the manpage and chunks
    # assert manpage and manpage.text
    # out_dir = MANSPLAIN_PATH / binary_name
    # out_dir.mkdir(exist_ok=True)
    # with (out_dir/manpage.id).with_suffix(".txt").open("w+") as f:
    #     f.write(f"Binary Name: {binary_name}\nId: {manpage.id}\nMetadata: {json.dumps(manpage.metadata)}\nLength: {len(manpage.text)}\nTokens: {chunker.get_chunk_size(manpage.text)}\nText:\n{manpage.text}")
    # for chunk in chunks:
    #     assert chunk.text
    #     with (out_dir/chunk.id).with_suffix(".txt").open("w+") as f:
    #         f.write(f"Binary Name: {binary_name}\nId: {chunk.id}\nMetadata: {json.dumps(chunk.metadata)}\nLength: {len(chunk.text)}\nTokens: {chunker.get_chunk_size(chunk.text)}\nText:\n{chunk.text}")
    
    # print(f"Manpage for {binary_name} loaded and chunked into {len(chunks)} chunks")
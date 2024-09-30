from unifai import UnifAIClient, tool
from unifai.wrappers.vector_db_clients._base_vector_db_client import VectorDBClient, VectorDBIndex
from _provider_defaults import PROVIDER_DEFAULTS

from pathlib import Path
GITA_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/geeda.json")
GITA_EMBEDDINGS_DB_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/gita_embeddings")

def make_gita_rag_prompt(index: VectorDBIndex, query: str, n_results: int = 50) -> str:
    result = index.query(query_texts=[query], n_results=n_results)[0]
    question_prompt = f"""Question: {query}\n\nContext:\n"""
    for metadata, document in zip(result.metadatas, result.documents):
        if document:
            question_prompt += f"Chapter {metadata['chapter_number']} {metadata['chapter']}, Line {metadata['line_number']}:\n"
            question_prompt += f"{document}\n\n"
    
    print(f"\033[34;1;4mRAG PROMPT:\n{question_prompt}\033[0m")
    return question_prompt


def gita_chat():
    ai = UnifAIClient(
            provider_client_kwargs={
                # "google": PROVIDER_DEFAULTS["google"][1],
                "openai": PROVIDER_DEFAULTS["openai"][1],
                "ollama": PROVIDER_DEFAULTS["ollama"][1],
                "chroma": PROVIDER_DEFAULTS["chroma"][1]
            }
        )
    
    gita_index = ai.get_or_create_index(        
        name="gita", 
        vector_db_provider="chroma",
        embedding_provider="openai",
        embedding_model="text-embedding-3-large"
    )

    system_prompt = """Your role is to answer questions as if you are Krishna using the Bhagavad Gita as a guide. You will be given a question and relevant context from the Gita. You must provide an answer based on the context."""
    chat = ai.chat(system_prompt=system_prompt)
    while True:
        query = input("\033[35m\nEnter question or CTRL+C to quit.\n>>> \033[0m")
        gita_rag_prompt = make_gita_rag_prompt(gita_index, query)
        for chunk in chat.send_message_stream(gita_rag_prompt):
            print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    gita_chat()
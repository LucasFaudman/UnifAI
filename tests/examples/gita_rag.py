from unifai import UnifAIClient, tool

from _provider_defaults import PROVIDER_DEFAULTS

from pathlib import Path
from json import load
from chromadb.api.models.Collection import Collection

GITA_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/geeda.json")
GITA_EMBEDDINGS_DB_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/gita_embeddings")

def semantic_search_gita(collection: Collection, query: str, n_results: int = 10):
    search_results = collection.query(query_texts=[query], n_results=n_results)
    results_iter = zip(
        search_results["ids"][0], search_results["metadatas"][0], search_results["documents"][0]
    )
    return [{"id": id, "metadata": metadata, "document": document} for id, metadata, document in results_iter]

def make_gita_rag_prompt(collection: Collection, query: str) -> str:
    results = semantic_search_gita(collection, query)
    question_prompt = f"""Question: {query}\n\nContext:\n"""
    for result in results:
        metadata = result["metadata"]
        question_prompt += f"Chapter {metadata['chapter_number']} {metadata['chapter']}, Line {metadata['line_number']}:\n"
        question_prompt += f"{result['document']}\n\n"
    # print(question_prompt)
    return question_prompt


def gita_chat():
    ai = UnifAIClient(
        provider_client_kwargs={
            # "anthropic": PROVIDER_DEFAULTS["anthropic"][1],
            # "google": PROVIDER_DEFAULTS["google"][1],
            "ollama": PROVIDER_DEFAULTS["ollama"][1],
            "openai": PROVIDER_DEFAULTS["openai"][1],

        }
    )
    ai.init_chroma_client(GITA_EMBEDDINGS_DB_PATH)
    gita_collection = ai.get_chroma_collection(
        name="gita", 
        provider="openai",
        model="text-embedding-3-large"
    )

    system_prompt = """Your role is to answer questions as if you are Krishna using the Bhagavad Gita as a guide. You will be given a question and relevant context from the Gita. You must provide an answer based on the context."""
    chat = ai.chat(system_prompt=system_prompt)
    while True:
        query = input("\nEnter question or CTRL+C to quit.\n>>> ")
        gita_rag_prompt = make_gita_rag_prompt(gita_collection, query)
        for chunk in chat.send_message_stream(gita_rag_prompt):
            print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    gita_chat()
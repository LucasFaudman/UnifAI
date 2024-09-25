from unifai import UnifAIClient, tool

from _provider_defaults import PROVIDER_DEFAULTS

from pathlib import Path
from json import load
from chromadb.api.models.Collection import Collection

GITA_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/geeda.json")
GITA_EMBEDDINGS_DB_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/gita_embeddings")

def chunk_gita(path: Path) -> list[dict]:
    with path.open() as f:
        gita_dict = load(f)

    gita_chunks = []
    for chapter_num, (chapter_name, chapter_text) in enumerate(gita_dict.items()):        
        for line_num, line in enumerate(filter(bool, chapter_text.split("\n"))):
            line_id = f"{chapter_num}.{line_num}"
            gita_chunks.append({
                "id": line_id,
                "metadata": {
                    "chapter": chapter_name, 
                    "chapter_number": chapter_num,
                    "line_number": line_num
                },
                "document": line
            })

    print(f"{len(gita_chunks)=}")
    print(gita_chunks[0])
    return gita_chunks


def embed_gita(collection: Collection, gita_chunks: list[dict]):
    ids, metadatas, documents = [], [], []
    for chunk in gita_chunks:
        ids.append(chunk["id"])
        metadatas.append(chunk["metadata"])
        documents.append(chunk["document"])

    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents
    )


def main():
    
    ai = UnifAIClient(
            provider_client_kwargs={
                "google": PROVIDER_DEFAULTS["google"][1],
                "openai": PROVIDER_DEFAULTS["openai"][1],
                "ollama": PROVIDER_DEFAULTS["ollama"][1]
            }
        )
    ai.init_chroma_client(GITA_EMBEDDINGS_DB_PATH)
    gita_collection = ai.get_chroma_collection(
        name="gita", 
        provider="openai",
        model="text-embedding-3-large")
    
    gita_chunks = chunk_gita(GITA_PATH)
    embed_gita(gita_collection, gita_chunks)



if __name__ == "__main__":
    main()
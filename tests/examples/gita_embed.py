from unifai import UnifAI
from _provider_defaults import PROVIDER_DEFAULTS

from pathlib import Path
from json import load

# https://prabhupadabooks.com/books
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


def main():
    
    ai = UnifAI(
            provider_client_kwargs={
                "google": PROVIDER_DEFAULTS["google"][1],
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
            
    ids, metadatas, documents = [], [], []
    for chunk in chunk_gita(GITA_PATH):
        ids.append(chunk["id"])
        metadatas.append(chunk["metadata"])
        documents.append(chunk["document"])

    gita_index.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=documents
    )


if __name__ == "__main__":
    main()
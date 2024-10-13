from unifai import UnifAIClient, tool
from unifai.types.rag_spec import RAGSpec, DEFAULT_RAG_PROMPT_TEMPLATE
from unifai.types import VectorDBQueryResult
from _provider_defaults import PROVIDER_DEFAULTS

import pygame
from pathlib import Path
GITA_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/geeda.json")
GITA_EMBEDDINGS_DB_PATH = Path("/Users/lucasfaudman/Documents/UnifAI/scratch/gita_embeddings")


ai = UnifAIClient(
    provider_client_kwargs={
        "openai": PROVIDER_DEFAULTS["openai"][1],
        "chroma": PROVIDER_DEFAULTS["chroma"][1],
        "nvidia": PROVIDER_DEFAULTS["nvidia"][1],
        "cohere": PROVIDER_DEFAULTS["cohere"][1],
        "google": PROVIDER_DEFAULTS["google"][1],
        "rank_bm25": PROVIDER_DEFAULTS["rank_bm25"][1],
    }
)

pygame.mixer.init()
def play_speech(text, voice="fable"):
    openai_client = ai.get_client("openai").client
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
    )
    outfile = "/Users/lucasfaudman/Documents/UnifAI/scratch/speech.mp3"
    response.stream_to_file(outfile)
    pygame.mixer.music.load(outfile)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # wait for music to finish playing
        pass


def format_gita_query_result(result: VectorDBQueryResult) -> str:
    formatted_result = ""
    for metadata, document in zip(result.metadatas, result.documents):
        if document:
            formatted_result += f"Chapter {metadata['chapter_number']} {metadata['chapter']}, Line {metadata['line_number']}:\n"
            formatted_result += f"{document}\n\n"
    return formatted_result

gita_prompt_template = DEFAULT_RAG_PROMPT_TEMPLATE
gita_prompt_template.value_formatters[VectorDBQueryResult] = format_gita_query_result


def gita_chat():
    rag_engine = ai.get_rag_engine(RAGSpec(
        index_name="gita",
        top_k=50,
        top_n=15,
        vector_db_provider="chroma",
        embedding_provider="openai",
        rerank_provider="rank_bm25",
        prompt_template=gita_prompt_template,
    ))

    system_prompt = """Your role is to answer questions as if you are Krishna using the Bhagavad Gita as a guide. You will be given a question and relevant context from the Gita. You must provide an answer based on the context."""
    opener = "Welcome to the Haree KrishnAI. I am Krishna. Ask me anything."
    print(f"\033[32;1m{opener}\n\033[0m")
    play_speech(opener, voice="fable")
    
    chat = ai.chat(system_prompt=system_prompt)
    while True:
        try:
            query = input("\033[34;1m\nEnter a question for \033[32;1mKrishna\033[34;1m or CTRL+C to switch provider.\n>>> \033[0m")
        except KeyboardInterrupt:
            new_provider = input("\033[35m\nEnter provider or CTRL+C to quit: \033[0m")
            chat.set_provider(new_provider)
            continue
        
        gita_rag_prompt = rag_engine.ragify(query)
        print(f"\033[34;1mRAG PROMPT:\n{gita_rag_prompt}\033[32;1m")
        
        stack = []
        for chunk in chat.send_message_stream(gita_rag_prompt):
            if not chunk.content: continue
            print(chunk.content, end="", flush=True)
            stack.append(chunk.content)
            if chunk.content.strip(" ").endswith((".", "?", "!", "\n")):
                    play_speech("".join(stack), voice="fable")
                    stack.clear()
        print("\033[0m")
        
if __name__ == "__main__":
    gita_chat()
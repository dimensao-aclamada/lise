# chatbot.py

import os
import json
import requests

from lise.config import GROQ_API_KEY, GROQ_MODEL
from lise.rag import RAGIndex

# Point to the directory with saved indexes and the master config file
INDEX_DIR = "rag_indexes"
CONFIG_FILE = "websites.json"


class GroqChatbot:
    def __init__(self, rag_index):
        self.rag = rag_index

    def generate_reply(self, query):
        """Generates a reply using the pre-loaded RAG context."""
        context_chunks = self.rag.retrieve(query)
        context = "\n\n".join(context_chunks)

        prompt = f"""Answer the user's question based on the context below.
Context:
{context}

User: {query}
Assistant:"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RuntimeError("Rate limit exceeded with Groq API.") from e
            raise


def answer_with_datasource(data_source: str, query: str) -> str:
    """
    Answers a query using a pre-indexed data source.
    This function is the core API-facing logic.
    """
    index_path = os.path.join(INDEX_DIR, f"{data_source}.index")
    chunks_path = os.path.join(INDEX_DIR, f"{data_source}_chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Data source '{data_source}' has not been indexed. "
            f"Run 'python manage.py crawl {data_source}' first."
        )

    rag = RAGIndex()
    rag.load_index(index_path, chunks_path)
    
    bot = GroqChatbot(rag)
    return bot.generate_reply(query)


def get_available_sources():
    """Lists available data sources from the configuration file."""
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    return list(config.keys())


def main():
    """
    Runs an interactive chatbot session against a pre-indexed data source.
    This function no longer handles crawling or website registration.
    """
    print("--- Groq Chatbot Interface ---")
    
    available_sources = get_available_sources()
    if not available_sources:
        print("\nNo data sources found in 'websites.json'.")
        print("Please add and crawl a data source first using the management script:")
        print("  python manage.py add <name> <url>")
        print("  python manage.py crawl <name>")
        return

    print("\nAvailable data sources:", ", ".join(available_sources))
    data_source = input("Enter the name of the data source to chat with: ").strip()

    if data_source not in available_sources:
        print(f"\nError: Data source '{data_source}' is not registered in '{CONFIG_FILE}'.")
        print("Use 'python manage.py add ...' to add it first.")
        return

    index_path = os.path.join(INDEX_DIR, f"{data_source}.index")
    if not os.path.exists(index_path):
        print(f"\nError: Index for '{data_source}' not found.")
        print(f"Please run 'python manage.py crawl {data_source}' to build the index.")
        return
        
    try:
        print(f"\nLoading index for '{data_source}'...")
        rag = RAGIndex()
        chunks_path = os.path.join(INDEX_DIR, f"{data_source}_chunks.json")
        rag.load_index(index_path, chunks_path)
        
        bot = GroqChatbot(rag)
        print("Chatbot is ready. Type 'exit' or 'quit' to end the session.")
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ("exit", "quit"):
                break
            try:
                reply = bot.generate_reply(query)
                print("\nBot:", reply)
            except Exception as e:
                print(f"\nAn error occurred while generating a reply: {e}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
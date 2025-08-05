# chatbot.py

import os
import json
import requests

from lise.config import GROQ_API_KEY, GROQ_MODEL
from lise.rag import RAGIndex

INDEX_DIR = "rag_indexes"
CONFIG_FILE = "websites.json"

class GroqChatbot:
    """
    A chatbot that uses a RAG index and can optionally maintain
    conversational history.
    """
    def __init__(self, rag_index, enable_history: bool = False):
        """
        Initializes the chatbot.

        Args:
            rag_index (RAGIndex): The loaded RAG index to retrieve context from.
            enable_history (bool): If True, the bot will remember past turns
                                   in the conversation. Defaults to False.
        """
        self.rag = rag_index
        self.enable_history = enable_history
        self.history = []

    def generate_reply(self, query: str) -> str:
        """
        Generates a reply using RAG context and optional conversation history.
        """
        # 1. Retrieve context using RAG, regardless of history
        context_chunks = self.rag.retrieve(query)
        context = "\n\n".join(context_chunks)

        # 2. Create the system prompt with instructions and RAG context
        system_prompt = f"""You are a helpful assistant. Answer the user's question based on the context provided below.
If the answer is not in the context, clearly state that you don't know.

Context:
{context}"""

        # 3. Build the message payload for the API
        messages = [{"role": "system", "content": system_prompt}]

        # 4. If history is enabled, add past messages
        if self.enable_history and self.history:
            messages.extend(self.history)
        
        # 5. Add the current user query
        messages.append({"role": "user", "content": query})

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": messages
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]

            # 6. If history is enabled, save the current exchange
            if self.enable_history:
                self.history.append({"role": "user", "content": query})
                self.history.append({"role": "assistant", "content": reply})

            return reply
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RuntimeError("Rate limit exceeded with Groq API.") from e
            raise

def answer_with_datasource(data_source: str, query: str, enable_history: bool = False) -> str:
    """
    Answers a query using a pre-indexed data source.
    This function is now the core of the API logic.
    """
    index_path = os.path.join(INDEX_DIR, f"{data_source}.index")
    chunks_path = os.path.join(INDEX_DIR, f"{data_source}_chunks.json")

    # Check for the data and raise a specific error if not found
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Data source '{data_source}' has not been indexed. "
            f"Run 'python manage.py crawl {data_source}' first."
        )
    # Load the persistent index and chunks
    rag = RAGIndex()
    rag.load_index(index_path, chunks_path)
    
    # Initialize the bot and generate a reply
    bot = GroqChatbot(rag, enable_history=enable_history)
    return bot.generate_reply(query)

def get_available_sources():
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


def main():
    """
    Runs an interactive chatbot session, now with conversational history enabled.
    """
    print("--- Groq Chatbot Interface ---")
    
    available_sources = get_available_sources()
    if not available_sources:
        print("\nNo data sources found. Use 'manage.py' to add and crawl one.")
        return

    print("\nAvailable data sources:", ", ".join(available_sources))
    data_source = input("Enter the data source to chat with: ").strip()

    if data_source not in available_sources:
        print(f"\nError: Data source '{data_source}' is not registered.")
        return

    try:
        index_path = os.path.join(INDEX_DIR, f"{data_source}.index")
        chunks_path = os.path.join(INDEX_DIR, f"{data_source}_chunks.json")

        print(f"\nLoading index for '{data_source}'...")
        rag = RAGIndex()
        rag.load_index(index_path, chunks_path)
        
        # Instantiate the bot with conversational history enabled for the interactive session
        bot = GroqChatbot(rag, enable_history=True)
        print("Chatbot is ready (with conversational history). Type 'exit' to quit.")
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ("exit", "quit"):
                break
            reply = bot.generate_reply(query)
            print("\nBot:", reply)

    except FileNotFoundError:
        print(f"\nError: Index for '{data_source}' not found.")
        print(f"Please run 'python manage.py crawl {data_source}' to build it.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
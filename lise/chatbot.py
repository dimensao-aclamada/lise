# chatbot.py

import os
import json
import requests

from lise.config import GROQ_API_KEY, GROQ_MODEL
from lise.rag import RAGIndex
from lise.crawler import crawl_website

CONFIG_FILE = "websites.json"
CHUNKS_DIR = "website_chunks"


INDEX_DIR = "rag_indexes" # Point to the directory with saved indexes

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
                # Re-raise as a specific, catchable error for the API
                raise RuntimeError("Rate limit exceeded with Groq API.") from e
            raise # Re-raise other HTTP errors

# This function is now the core of the API logic
def answer_with_datasource(data_source: str, query: str) -> str:
    """
    Answers a query using a pre-indexed data source.
    This function NO LONGER crawls or creates files.
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
    bot = GroqChatbot(rag)
    return bot.generate_reply(query)

def load_websites_config():
    """
    Loads or initializes the websites.json configuration file.

    Returns:
        dict: Parsed config from JSON.
    """
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def chunks_file_path(data_source_key):
    """
    Returns the expected path for a chunk file given a data source name.
    
    Args:
        data_source_key (str): Logical name of the data source.

    Returns:
        str: Path to the corresponding JSON file with page chunks.
    """
    return os.path.join(CHUNKS_DIR, f"{data_source_key}_chunks.json")

def save_websites_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def main():
    websites = load_websites_config()

    data_source = input("Enter the data source name (e.g., mysitefaster): ").strip()

    if data_source not in websites:
        print(f"Data source '{data_source}' is not registered.")
        website = input("Enter the website URL (e.g., https://example.com): ").strip()
        if not website.startswith(("http://", "https://")):
            website = "https://" + website

        instructions = input("Optional: Add crawling instructions (or press Enter to skip): ").strip()
        instructions = instructions or "Default crawl. No specific instruction."

        mandatory_pages = ["/"]
        exclude_pages = []

        # Store only domain as `website` key
        clean_domain = website.replace("https://", "").replace("http://", "").rstrip("/")

        websites[data_source] = {
            "website": clean_domain,
            "mandatory_pages": mandatory_pages,
            "exclude_pages": exclude_pages,
            "instructions": instructions
        }

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(websites, f, ensure_ascii=False, indent=2)

        print(f"Registered '{data_source}' in {CONFIG_FILE}.")

    config = websites[data_source]
    base_url = config["website"]
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url

    chunks_path = chunks_file_path(data_source)

    if os.path.exists(chunks_path):
        print(f"Chunks already exist at '{chunks_path}'. Skipping crawl.")
        with open(chunks_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
    else:
        print(f"Crawling {base_url} with instructions: {config['instructions']}")
        pages = crawl_website(
            base_url=base_url,
            must_include=config.get("mandatory_pages", []),
            must_exclude=config.get("exclude_pages", [])
        )
        print(f"{len(pages)} pages retrieved.")

        if not pages:
            print("Error: No valid pages retrieved. Exiting.")
            return

        os.makedirs(CHUNKS_DIR, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

    rag = RAGIndex()
    rag.build_index(pages)
    bot = GroqChatbot(rag)

    print("\nChatbot is ready. Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ("exit", "quit"):
            break
        reply = bot.generate_reply(query)
        print("\nBot:", reply)


if __name__ == "__main__":
    #print(answer_with_datasource("mysitefaster","what is the main service?"))
    main()
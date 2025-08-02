# chatbot.py

import os
import json
import requests
from config import GROQ_API_KEY, GROQ_MODEL
from lise.lise.rag import RAGIndex
from lise.lise.crawler import crawl_website


CHUNKS_DIR = "website_chunks"
CONFIG_FILE = "lise/websites.json"

class GroqChatbot:
    def __init__(self, rag_index=None):
        """
        Inicializa o chatbot com ou sem um índice RAG.

        Args:
            rag_index (RAGIndex | None): Instância de RAGIndex ou None se não for usado contexto.
        """
        self.rag = rag_index
        self.history = []

    def generate_reply(self, query):
        """
        Gera uma resposta a partir do modelo Groq com ou sem contexto adicional.

        Args:
            query (str): A pergunta do utilizador.

        Returns:
            str: Resposta gerada pelo modelo.
        """
        if self.rag:
            context_chunks = self.rag.retrieve(query)
            context = "\n\n".join(context_chunks)
            prompt = f"""Responde à pergunta do utilizador com base no contexto abaixo.
Contexto:
{context}

Utilizador: {query}
Assistente:"""
        else:
            prompt = f"""Responde de forma completa e útil à pergunta abaixo.
Utilizador: {query}
Assistente:"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        return reply

def load_websites_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def chunks_file_path(key):
    return os.path.join(CHUNKS_DIR, f"{key}_chunks.json")


def main():
    websites = load_websites_config()

    url = input("Indica a URL base do website (ex: aimaggie.com): ").strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    domain = url.replace("https://", "").replace("http://", "").strip("/")

    if domain not in websites:
        print(f"O website '{domain}' não está registado em {CONFIG_FILE}.")
        return

    config = websites[domain]
    key = config["key"]
    chunks_path = chunks_file_path(key)

    if os.path.exists(chunks_path):
        print(f"Chunks já existentes em '{chunks_path}'. A saltar rastreio.")
        with open(chunks_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
    else:
        print(f"Crawling {domain} com instruções: {config['instructions']}")
        pages = crawl_website(
            base_url=url,
            must_include=config["mandatory_pages"],
            must_exclude=config["exclude_pages"]
        )
        print(f"{len(pages)} páginas obtidas.")
        os.makedirs(CHUNKS_DIR, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

    rag = RAGIndex()
    rag.build_index(pages)
    bot = GroqChatbot(rag)

    print("\nChatbot pronto. Escreve 'exit' para sair.")
    while True:
        query = input("\nTu: ")
        if query.lower() in ("exit", "quit"):
            break
        reply = bot.generate_reply(query)
        print("\nBot:", reply)


if __name__ == "__main__":
    main()
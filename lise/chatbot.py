# chatbot.py

import requests
import json
from lise.lise.config import GROQ_API_KEY, GROQ_MODEL
from lise.lise.rag import RAGIndex
from lise.lise.crawler import crawl_website


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


def main():
    print("Escolhe a fonte de contexto:")
    print("1. Website")
    print("2. Nenhum")
    context_choice = input("Opção [1-2]: ").strip()

    if context_choice == "1":
        url = input("Indica a URL base do website a rastrear: ").strip()
        print(f"Rastreando {url}...")
        pages = crawl_website(url)
        print(f"Foram encontradas {len(pages)} páginas.")
        rag = RAGIndex()
        rag.build_index(pages)
    elif context_choice == "2":
        rag = None
    else:
        print("Opção inválida. A terminar.")
        return

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
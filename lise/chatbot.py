# lise/chatbot.py (Refactored Version)

import requests
import json
from lise.config import GROQ_MODEL # We still need the model name

class GroqChatbot:
    """
    A chatbot that uses a RAG index and can optionally maintain
    conversational history. It is initialized with the platform API key.
    """
    def __init__(self, rag_index, platform_api_key: str, enable_history: bool = False):
        """
        Initializes the chatbot.

        Args:
            rag_index (RAGIndex): The loaded RAG index to retrieve context from.
            platform_api_key (str): The specific API key for the LLM platform (e.g., Groq).
            enable_history (bool): If True, remembers past turns in the conversation.
        """
        self.rag = rag_index
        self.platform_api_key = platform_api_key
        self.enable_history = enable_history
        self.history = []

    def generate_reply(self, query: str) -> str:
        """
        Generates a reply using RAG context and optional conversation history.
        """
        context_chunks = self.rag.retrieve(query)
        context = "\n\n".join(context_chunks)

        system_prompt = f"""You are a helpful assistant. Answer the user's question based on the context provided below.
If the answer is not in the context, clearly state that you don't know.

Context:
{context}"""

        messages = [{"role": "system", "content": system_prompt}]
        if self.enable_history and self.history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": query})

        headers = {
            # Use the API key provided during initialization
            "Authorization": f"Bearer {self.platform_api_key}",
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

            if self.enable_history:
                self.history.append({"role": "user", "content": query})
                self.history.append({"role": "assistant", "content": reply})

            return reply
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RuntimeError("Rate limit exceeded with Groq API.") from e
            raise
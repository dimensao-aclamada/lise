# lise/__init__.py (Refactored Version)

"""
A package for building and querying a RAG-based chatbot with a database backend.
"""

# Expose the core, reusable classes of the package.
from .chatbot import GroqChatbot
from .rag import RAGIndex
from .crawler import crawl_website
from .encryption import encrypt_key, decrypt_key

# Define what symbols are imported when a user does 'from lise import *'
__all__ = [
    "GroqChatbot",
    "RAGIndex",
    "crawl_website",
    "encrypt_key",
    "decrypt_key",
]
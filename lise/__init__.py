# lise/lise/__init__.py

"""
A package for building and querying a RAG-based chatbot
with a Groq backend.
"""

# Expose the primary, high-level function for easy use.
# This is the main entry point for the API.
from .chatbot import answer_with_datasource

# Expose the core classes for users who might need more direct control
# over the indexing or chatbot instantiation process.
from .rag import RAGIndex
from .chatbot import GroqChatbot

# Define what symbols are imported when a user does 'from lise import *'
# This is a best practice for defining the public API of a package.
__all__ = [
    "answer_with_datasource",
    "RAGIndex",
    "GroqChatbot",
]
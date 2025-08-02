# config.py
import os
from dotenv import load_dotenv

GROQ_MODEL = "llama3-8b-8192"
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
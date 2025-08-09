# lise/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
# The single source of truth for the database file path.
DATABASE_FILE = "lise.db"

# --- Encryption Configuration ---
# The master key for encrypting/decrypting third-party API keys stored in the database.
# This MUST be set in your .env file for the application to run.
LISE_ENCRYPTION_KEY = os.getenv("LISE_ENCRYPTION_KEY")

# Fail fast if the encryption key is not set.
if not LISE_ENCRYPTION_KEY:
    raise RuntimeError(
        "FATAL: LISE_ENCRYPTION_KEY is not set in the .env file. "
        "Please generate a key and add it to your .env file to proceed."
    )

# --- LLM Model Configuration ---
# Default model to be used if not otherwise specified.
GROQ_MODEL = "llama3-8b-8192"

# --- RAG Configuration ---
# These settings control the text chunking and embedding process.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
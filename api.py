# api.py

import sqlite3
import uuid
import os
import requests
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, status, Header
from pydantic import BaseModel
from typing import Optional, List

# --- Lise Imports ---
from lise.config import DATABASE_FILE, EMBED_MODEL
from lise.encryption import decrypt_key
from lise.rag import RAGIndex
from lise.chatbot import GroqChatbot
from sentence_transformers import SentenceTransformer

# --- In-Memory Session & Model Store ---
chatbots = {} # For persistent conversations
# Load the embedding model once at startup to reuse it for in-memory requests
embedding_model = SentenceTransformer(EMBED_MODEL)

# --- Data Models for API ---
class AnswerRequest(BaseModel):
    query: str
    chunks_url: Optional[str] = None # <-- ADDED: The new optional parameter
    conversation_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    conversation_id: str
    property_id: int

# --- NEW: Helper Class for On-the-Fly RAG ---
class InMemoryRAG:
    """A temporary RAG index built from chunks fetched from a URL."""
    def __init__(self, chunks: List[str], model: SentenceTransformer):
        if not chunks:
            raise ValueError("Cannot initialize InMemoryRAG with empty chunks.")
        
        self.chunks = chunks
        self.model = model
        
        # Create a temporary FAISS index in memory
        print(f"-> Building in-memory index for {len(chunks)} chunks...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=False) # No progress bar in API
        embeddings_np = np.array(embeddings).astype("float32")
        
        self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        print("-> In-memory index built.")

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieves chunks from the in-memory index."""
        q_emb = self.model.encode([query], show_progress_bar=False)
        q_emb_np = np.array(q_emb).astype("float32")
        
        _, I = self.index.search(q_emb_np, top_k)
        
        return [self.chunks[i] for i in I[0]]

# --- Database & Property Lookup (Unchanged) ---
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def get_property_from_api_key(api_key: str, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM properties WHERE lise_api_key = ?", (api_key,))
    return cursor.fetchone()

# --- FastAPI App ---
app = FastAPI(
    title="Lise Chatbot API",
    description="An API for chatting with RAG-indexed data sources."
)

@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(
    request: AnswerRequest,
    x_api_key: str = Header(..., description="Your unique Lise API key for the property.")
):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        # 1. Authenticate and get property details (ALWAYS REQUIRED)
        prop = get_property_from_api_key(x_api_key, conn)
        if not prop:
            raise HTTPException(status_code=401, detail="Invalid Lise API Key.")
        
        property_id = prop['id']
        platform_api_key = decrypt_key(prop['platform_api_key'])

        # 2. Check for chunks_url to decide the workflow
        if request.chunks_url:
            # --- STATELESS, ON-THE-FLY WORKFLOW ---
            print(f"Handling on-the-fly request from URL for Property ID {property_id}")
            try:
                # Download chunks from the provided URL
                response = requests.get(request.chunks_url, timeout=10)
                response.raise_for_status()
                data = response.json() # Get the raw JSON data
                
                # --- THIS IS THE CRUCIAL NEW LOGIC ---
                # Check if the data is a list of objects (our export format)
                if isinstance(data, list) and data and isinstance(data[0], dict) and 'chunk_text' in data[0]:
                    # This looks like our export format, so we extract the 'chunk_text' field
                    chunks = [item['chunk_text'] for item in data if 'chunk_text' in item]
                    print(f"-> Parsed {len(chunks)} chunks from structured JSON object.")
                elif isinstance(data, list) and all(isinstance(c, str) for c in data):
                    # This is for the simple list of strings format
                    chunks = data
                    print(f"-> Parsed {len(chunks)} chunks from simple JSON array.")
                else:
                    # The format is invalid
                    raise HTTPException(status_code=400, detail="URL did not return a valid JSON array of strings or the expected chunk object format.")
                # --- END OF CRUCIAL NEW LOGIC ---

                # Now that we have the chunks, the rest of the process is the same
                in_memory_rag = InMemoryRAG(chunks, embedding_model)
                bot = GroqChatbot(
                    rag_index=in_memory_rag,
                    platform_api_key=platform_api_key,
                    enable_history=False
                )
                answer = bot.generate_reply(request.query)
                
                return AnswerResponse(
                    answer=answer,
                    conversation_id=str(uuid.uuid4()),
                    property_id=property_id
                )

            except requests.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch chunks from URL: {e}")
            except Exception as e:
                print(f"An unexpected error occurred during on-the-fly processing: {e}")
                raise HTTPException(status_code=500, detail="Failed to process chunks from URL.")

    finally:
        if conn:
            conn.close()

# --- Main entry point (Unchanged) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Lise API server...")
    uvicorn.run("api:app", host="0.0.0.0", port=5001, reload=True)
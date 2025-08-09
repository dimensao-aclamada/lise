# api.py (Updated for per-source keys)

import uuid
import json
import os
import secrets  # Required for timing-safe key comparison
from fastapi import FastAPI, HTTPException, status, Header
from pydantic import BaseModel
from typing import Optional

from lise.chatbot import GroqChatbot, RAGIndex
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Still needed for GROQ_API_KEY inside the chatbot module
CONFIG_FILE = "websites.json"
INDEX_DIR = "rag_indexes"

# --- In-Memory Session Store ---
chatbots = {}

# --- Data Models ---
class AnswerRequest(BaseModel):
    data_source: str
    query: str
    conversation_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    conversation_id: str

# --- FastAPI App ---
app = FastAPI(
    title="Lise Chatbot API",
    description="An API for chatting with RAG-indexed data sources."
)

def get_config():
    """Helper to load the websites config."""
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(
    request: AnswerRequest,
    x_api_key: str = Header(..., description="The API key for the specific data source.")
):
    """
    Get an answer from a data source. The API key must match the one
    assigned to the requested data_source.
    """
    config = get_config()
    data_source_name = request.data_source
    
    # --- Security Validation ---
    if data_source_name not in config:
        raise HTTPException(status_code=404, detail=f"Data source '{data_source_name}' not found.")
        
    source_config = config[data_source_name]
    expected_key = source_config.get("api_key")
    
    if not expected_key or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API Key for this data source.")

    # --- Main Logic ---
    try:
        conversation_id = request.conversation_id
        if conversation_id and conversation_id in chatbots:
            bot = chatbots[conversation_id]
        else:
            conversation_id = str(uuid.uuid4())
            index_path = os.path.join(INDEX_DIR, f"{data_source_name}.index")
            if not os.path.exists(index_path):
                raise FileNotFoundError() # Caught below

            chunks_path = os.path.join(INDEX_DIR, f"{data_source_name}_chunks.json")
            rag = RAGIndex()
            rag.load_index(index_path, chunks_path)
            bot = GroqChatbot(rag, enable_history=True)
            chatbots[conversation_id] = bot

        answer = bot.generate_reply(request.query)
        return AnswerResponse(answer=answer, conversation_id=conversation_id)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Index for '{data_source_name}' not found. Please crawl it first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") # For server-side debugging
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
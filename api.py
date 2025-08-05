# api.py (Corrected Version)

import uuid
import json
import os
from fastapi import FastAPI, HTTPException, Depends, status, Header
from pydantic import BaseModel
from typing import Optional

from lise.chatbot import GroqChatbot, RAGIndex
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()
API_KEY = os.getenv("MY_API_KEY")
if not API_KEY:
    raise RuntimeError("MY_API_KEY is not set in the .env file.")

CONFIG_FILE = "websites.json"
INDEX_DIR = "rag_indexes"

# --- In-Memory Session Store ---
# Warning: This is for demonstration only. It will not work across multiple server processes.
chatbots = {}

# --- Security Dependency ---
# This is the corrected dependency function.
# It now explicitly looks for a header named "X-API-Key".
async def verify_api_key(x_api_key: str = Header(..., description="dcdsc897sdc87sd9c87sd9c87s9d8csdhc9ds8c8")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )

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

@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(
    request: AnswerRequest,
    # The dependency is now a parameter of the function itself, which is cleaner.
    _api_key: None = Depends(verify_api_key)
):
    """
    Get an answer from a data source. Start or continue a conversation.
    """
    try:
        data_source = request.data_source
        query = request.query
        conversation_id = request.conversation_id

        if conversation_id and conversation_id in chatbots:
            # Continue existing conversation
            bot = chatbots[conversation_id]
        else:
            # Start a new conversation
            conversation_id = str(uuid.uuid4())
            print(f"Starting new conversation: {conversation_id}")

            index_path = os.path.join(INDEX_DIR, f"{data_source}.index")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Data source '{data_source}' has not been indexed.")

            chunks_path = os.path.join(INDEX_DIR, f"{data_source}_chunks.json")
            rag = RAGIndex()
            rag.load_index(index_path, chunks_path)
            
            # Create a new bot with history enabled and store it
            bot = GroqChatbot(rag, enable_history=True)
            chatbots[conversation_id] = bot

        # Generate the answer
        answer = bot.generate_reply(query)

        return AnswerResponse(answer=answer, conversation_id=conversation_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        # It's good practice to log the actual error for debugging
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An internal server error occurred."
        )

# Optional: Endpoint to list available data sources
@app.get("/api/sources", response_model=list[str])
async def get_available_data_sources(_api_key: None = Depends(verify_api_key)):
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    return list(config.keys())
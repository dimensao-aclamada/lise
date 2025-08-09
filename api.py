# api.py

import sqlite3
import uuid
import os
from fastapi import FastAPI, HTTPException, status, Header, Depends
from pydantic import BaseModel
from typing import Optional

# --- Lise Imports ---
from lise.config import DATABASE_FILE
from lise.encryption import decrypt_key
from lise.rag import RAGIndex
from lise.chatbot import GroqChatbot

# --- In-Memory Session Store ---
# Warning: This is for demonstration only. It will not work across multiple server processes.
# For production, a Redis or similar cache would be needed.
chatbots = {}

# --- Data Models for API ---
class AnswerRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    conversation_id: str
    property_id: int

# --- Database & Property Lookup ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False) # check_same_thread is needed for FastAPI
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def get_property_from_api_key(api_key: str, conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Finds a property record in the database using the Lise API key."""
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
    """
    Receives a query, authenticates the property via its API key,
    and returns a conversational response.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        # 1. Authenticate and Authorize
        # Find the property associated with the provided Lise API key.
        prop = get_property_from_api_key(x_api_key, conn)
        if not prop:
            raise HTTPException(status_code=401, detail="Invalid Lise API Key.")
        
        property_id = prop['id']
        
        # 2. Handle Conversational State
        conversation_id = request.conversation_id
        if conversation_id and conversation_id in chatbots:
            # Continue existing conversation
            bot = chatbots[conversation_id]
        else:
            # Start a new conversation
            conversation_id = str(uuid.uuid4())
            print(f"Starting new conversation ({conversation_id}) for Property ID {property_id}")

            # Decrypt the platform API key stored for this property
            try:
                platform_api_key = decrypt_key(prop['platform_api_key'])
            except Exception:
                # This could happen if the master encryption key changed, for example
                raise HTTPException(status_code=500, detail="Could not decrypt platform API key.")
            
            # Instantiate the core RAG and Chatbot classes
            rag_index = RAGIndex(property_id=property_id)
            bot = GroqChatbot(
                rag_index=rag_index,
                platform_api_key=platform_api_key,
                enable_history=True
            )
            chatbots[conversation_id] = bot

        # 3. Generate the Reply
        answer = bot.generate_reply(request.query)

        # 4. Return the response
        return AnswerResponse(
            answer=answer,
            conversation_id=conversation_id,
            property_id=property_id
        )

    except Exception as e:
        # For security, don't leak detailed internal errors to the client.
        # Log the actual error for debugging.
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        if conn:
            conn.close()

# --- Main entry point for running the API server ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Lise API server...")
    # This allows running the API directly with `python api.py`
    uvicorn.run("api:app", host="0.0.0.0", port=5001, reload=True)

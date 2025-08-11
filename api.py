# api.py

import sqlite3
import uuid
import os
from fastapi import FastAPI, HTTPException, status, Header
from pydantic import BaseModel
from typing import Optional

# --- Lise Imports ---
from lise.config import DATABASE_FILE
from lise.encryption import decrypt_key
from lise.rag import RAGIndex
from lise.chatbot import GroqChatbot

# --- In-Memory Session Store ---
# This remains for managing multi-turn conversations for authenticated users.
chatbots = {}

# --- Data Models for API ---
class AnswerRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None # The user can only provide a query and a conversation_id

class AnswerResponse(BaseModel):
    answer: str
    conversation_id: str
    property_id: int

# --- Database & Property Lookup (Unchanged) ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        # check_same_thread is needed for FastAPI with SQLite
        conn = sqlite3.connect(DATABASE_FILE, check_same_thread=False) 
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
    description="An API for chatting with pre-indexed data sources."
)

@app.post("/api/answer", response_model=AnswerResponse)
async def get_answer(
    request: AnswerRequest,
    x_api_key: str = Header(..., description="Your unique Lise API key for the property.")
):
    """
    Receives a query, authenticates the property via its API key, and returns
    a conversational response based on its pre-indexed data.
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        # 1. Authenticate and Authorize
        # This is the security gate. We find the property tied to the key.
        prop = get_property_from_api_key(x_api_key, conn)
        if not prop:
            raise HTTPException(status_code=401, detail="Invalid Lise API Key.")
        
        property_id = prop['id']
        
        # 2. Handle Conversational State (No more on-the-fly logic)
        conversation_id = request.conversation_id
        if conversation_id and conversation_id in chatbots:
            # Continue existing conversation
            bot = chatbots[conversation_id]
        else:
            # Start a new persistent conversation
            conversation_id = str(uuid.uuid4())
            print(f"Starting new persistent conversation ({conversation_id}) for Property ID {property_id}")

            # Decrypt the platform API key stored for this specific property
            platform_api_key = decrypt_key(prop['platform_api_key'])
            
            # Instantiate the RAG and Chatbot classes using the authenticated property_id
            rag_index = RAGIndex(property_id=property_id)
            bot = GroqChatbot(
                rag_index=rag_index,
                platform_api_key=platform_api_key,
                enable_history=True
            )
            chatbots[conversation_id] = bot

        # 3. Generate the Reply using data only from the authenticated property
        answer = bot.generate_reply(request.query)

        # 4. Return the response
        return AnswerResponse(
            answer=answer,
            conversation_id=conversation_id,
            property_id=property_id
        )

    except FileNotFoundError as e:
        # This error is specifically for when the index file is missing
        print(f"Indexing error for Property ID {property_id}: {e}")
        raise HTTPException(status_code=404, detail="The data for this property has not been indexed yet. Please run the indexing process.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    finally:
        if conn:
            conn.close()

# --- Main entry point (Unchanged) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Lise API server...")
    uvicorn.run("api:app", host="0.0.0.0", port=5001, reload=True)
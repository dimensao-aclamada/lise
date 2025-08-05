# api.py

import os
import json
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from starlette.middleware.wsgi import WSGIMiddleware # Import WSGIMiddleware

# Load environment variables from .env file
load_dotenv()

# Import configurations and core modules
from lise.config import GROQ_API_KEY, GROQ_MODEL, MY_API_KEY
from lise.rag import RAGIndex
from lise.crawler import crawl_website # Though not directly used by the API endpoint, it's part of the original context

# Configuration constants
CONFIG_FILE = "websites.json"
CHUNKS_DIR = "website_chunks"
INDEX_DIR = "rag_indexes" # Point to the directory with saved indexes

# Initialize Flask application
flask_app = Flask(__name__) # Renamed to avoid conflict with 'app' for Uvicorn

# Wrap Flask app with WSGIMiddleware for Uvicorn
# This 'app' will be the one Uvicorn serves, ensuring it's treated as a WSGI application.
app = WSGIMiddleware(flask_app) 

class GroqChatbot:
    """
    Handles interactions with the Groq API for generating replies
    based on retrieved context.
    """
    def __init__(self, rag_index):
        self.rag = rag_index

    def generate_reply(self, query):
        """
        Generates a reply using the pre-loaded RAG context by calling the Groq API.
        """
        context_chunks = self.rag.retrieve(query)
        context = "\n\n".join(context_chunks)

        prompt = f"""Answer the user's question based on the context below.
Context:
{context}

User: {query}
Assistant:"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Re-raise as a specific, catchable error for the API consumer
                raise RuntimeError("Rate limit exceeded with Groq API. Please try again later.") from e
            # For other HTTP errors, re-raise the original exception
            raise 
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Network error connecting to Groq API: {e}") from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Groq API request timed out: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during API call
            raise RuntimeError(f"An unexpected error occurred during Groq API call: {e}") from e

def answer_with_datasource(data_source: str, query: str) -> str:
    """
    Answers a query using a pre-indexed data source.
    This function loads the RAG index and uses the GroqChatbot to generate a reply.
    """
    index_path = os.path.join(INDEX_DIR, f"{data_source}.index")
    chunks_path = os.path.join(INDEX_DIR, f"{data_source}_chunks.json")

    # Check for the data and raise a specific error if not found
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"Data source '{data_source}' has not been indexed. "
            f"Please ensure the index and chunks files exist at '{index_path}' and '{chunks_path}'."
        )
    
    # Load the persistent index and chunks
    rag = RAGIndex()
    rag.load_index(index_path, chunks_path)
    
    # Initialize the bot and generate a reply
    bot = GroqChatbot(rag)
    return bot.generate_reply(query)

# --- API Endpoint Definition ---

@flask_app.route("/api/answer", methods=["POST"]) # Use flask_app here for routing
def api_answer_query():
    """
    API endpoint to answer a query using a specified data source.
    Requires an API key for authentication.
    """
    # API Key Authentication. MY_API_KEY is loaded from .env in config.py.
    if MY_API_KEY:
        if request.headers.get("X-API-Key") != MY_API_KEY:
            return jsonify({"error": "Unauthorized: Invalid or missing API Key."}), 401
    else:
        print("Warning: MY_API_KEY is not set. API key authentication is disabled.")
    
    # Request Validation
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    data_source = data.get("data_source")
    query = data.get("query")

    if not data_source or not isinstance(data_source, str):
        return jsonify({"error": "Missing or invalid 'data_source' parameter. Must be a string."}), 400
    if not query or not isinstance(query, str):
        return jsonify({"error": "Missing or invalid 'query' parameter. Must be a string."}), 400

    # Core Logic Execution and Error Handling
    try:
        reply = answer_with_datasource(data_source, query)
        return jsonify({"reply": reply}), 200
    except FileNotFoundError as e:
        # Specific error for missing data source index
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:
        # Catch errors related to Groq API (e.g., rate limits, network issues)
        return jsonify({"error": str(e)}), 503 # Service Unavailable
    except Exception as e:
        # Catch any other unexpected server-side errors
        print(f"An unhandled error occurred: {e}") 
        return jsonify({"error": "Internal server error. Please try again later."}), 500

# This block is for direct execution of the Flask app, not typically used with Uvicorn
# if __name__ == "__main__":
#     flask_app.run(debug=True, host='0.0.0.0', port=5000)

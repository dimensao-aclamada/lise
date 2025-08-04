# api.py

from fastapi import FastAPI, Query, HTTPException
from lise.chatbot import answer_with_datasource

app = FastAPI(
    title="Chatbot API",
    version="1.0",
    description="API to chat with data from pre-indexed websites."
)

@app.get("/ask")
def ask(
    data_source: str = Query(..., description="The key of the pre-indexed data source (e.g., 'mysite')."),
    query: str = Query(..., description="The user's natural language question.")
):
    """
    Answers a query using a pre-indexed data source.
    """
    try:
        response = answer_with_datasource(data_source, query)
        return {"response": response}
    except FileNotFoundError as e:
        # If the index doesn't exist, return a helpful 404 error
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Handle the rate-limiting error from the chatbot
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        # Generic error for other issues
        print(f"An unexpected error occurred: {e}") # Log the error for debugging
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
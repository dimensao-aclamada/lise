# api.py

# File header
# api.py

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from lise.chatbot import answer_with_datasource

app = FastAPI(title="Chatbot API", version="1.0")

@app.get("/ask")
def ask(
    data_source: str = Query(..., description="The registered data source key"),
    query: str = Query(..., description="User's natural language question")
):
    """
    Read-only chatbot API to answer queries using a registered data source.
    
    Parameters:
    - data_source: Key in websites.json (e.g., 'mysitefaster')
    - query: The natural language question

    Returns:
    - JSON with assistant's reply
    """
    try:
        return {"response": answer_with_datasource(data_source, query)}
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
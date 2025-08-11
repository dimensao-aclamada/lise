# lise/rag.py

import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import List

from lise.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, DATABASE_FILE

# --- Constants ---
INDEX_DIR = "rag_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Database Helper ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error in RAG module: {e}")
        return None

# --- Smart Text Chunking Function ---
def recursive_character_splitter(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = None
) -> List[str]:
    """
    Splits text recursively based on a list of separators.
    Tries to split by the first separator, then the second, and so on,
    to keep semantically related text together.
    """
    if separators is None:
        # The order of separators is crucial: from largest semantic unit to smallest.
        separators = ["\n\n", "\n", ". ", " ", ""]

    if not text:
        return []

    # Find the first separator that exists in the text
    separator_to_use = ""
    for sep in separators:
        if sep == "": # Character-level split is the last resort
            separator_to_use = sep
            break
        if sep in text:
            separator_to_use = sep
            break
            
    # Split the text by the chosen separator
    if separator_to_use:
        splits = text.split(separator_to_use)
    else: # If no separators found, treat the whole text as one split
        splits = [text]

    # Process the splits to form chunks
    chunks = []
    current_chunk = ""
    for part in splits:
        if not part:
            continue
        
        # Add the separator back to the part (except for the first part of the split)
        if current_chunk:
             part_to_add = separator_to_use + part
        else:
            part_to_add = part
            
        # If adding the new part fits, append it to the current chunk
        if len(current_chunk) + len(part_to_add) <= chunk_size:
            current_chunk += part_to_add
        else:
            # If the current chunk is not empty, it's a complete chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # If the new part itself is too large, recursively split it
            if len(part_to_add) > chunk_size:
                chunks.extend(
                    recursive_character_splitter(part_to_add, chunk_size, chunk_overlap, separators[1:])
                )
            # Otherwise, the new part becomes the start of the next chunk
            else:
                current_chunk = part_to_add
                
    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Handle overlap (a simplified approach)
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk_end = chunks[i-1][-chunk_overlap:]
            overlapped_chunks.append(prev_chunk_end + chunks[i])
        return overlapped_chunks
        
    return chunks

# --- RAG Class ---
class RAGIndex:
    """
    A RAG system that loads its index from disk and retrieves chunk text
    from a URL stored in a database.
    """
    def __init__(self, property_id: int, model_name=EMBED_MODEL):
        if not isinstance(property_id, int):
            raise TypeError("property_id must be an integer.")
            
        self.property_id = property_id
        self.index_path = os.path.join(INDEX_DIR, f"{self.property_id}.index")
        self.model = SentenceTransformer(model_name)
        self.index = None

    def build_index_from_chunks(self, chunks: List[str]):
        """
        Takes a list of text chunks, creates a FAISS index from their embeddings,
        and saves the index file to disk. This method does NOT interact with the database.
        """
        if not chunks:
            print("Cannot build index from empty list of chunks.")
            return
            
        print("-> Encoding text chunks for vector index...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        faiss.write_index(self.index, self.index_path)
        print(f"-> Index built and saved to '{self.index_path}'.")

    def _load_index(self):
        """Loads the FAISS index from disk if it's not already loaded."""
        if self.index is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Index file not found for property {self.property_id} at {self.index_path}.")
            self.index = faiss.read_index(self.index_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Searches for a query, gets the relevant chunk URLs from the DB,
        downloads the chunks from that URL, and returns the relevant text.
        """
        self._load_index()

        # 1. Get relevant chunk indices from FAISS (0-based)
        q_emb = self.model.encode([query]).astype('float32')
        _, I = self.index.search(q_emb, top_k)
        faiss_indices = [int(i) for i in I[0]]

        # 2. Get the URL to the chunks JSON file from the database
        conn = get_db_connection()
        if not conn: return []
            
        try:
            cursor = conn.cursor()
            # For this architecture, we assume one datasource contains the relevant URL.
            # A more complex system might need to aggregate from multiple datasources.
            cursor.execute(
                "SELECT chunks_json_url FROM datasources WHERE property_id = ? AND status = 'completed' AND chunks_json_url IS NOT NULL LIMIT 1",
                (self.property_id,)
            )
            result = cursor.fetchone()
            if not result or not result['chunks_json_url']:
                raise ValueError("No indexed chunk URL found for this property in the database.")
            
            chunks_url = result['chunks_json_url']
            
            # 3. Download the entire list of chunks from the URL
            print(f"-> Retrieving chunks from {chunks_url}...")
            if chunks_url.startswith("file://"):
                # Handle local file URLs
                with open(chunks_url.replace("file://", ""), "r", encoding="utf-8") as f:
                    all_chunks = json.load(f)
            else:
                # Handle remote http/https URLs
                response = requests.get(chunks_url, timeout=10)
                response.raise_for_status()
                all_chunks = response.json()
            
            # 4. Return the specific chunks based on the indices found by FAISS
            retrieved_chunks = [all_chunks[i] for i in faiss_indices if i < len(all_chunks)]
            return retrieved_chunks
            
        except Exception as e:
            print(f"❌ An error occurred during retrieval: {e}")
            return []
        finally:
            if conn:
                conn.close()
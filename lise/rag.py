# lise/rag.py (Refactored Version)

import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from lise.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, DATABASE_FILE

# --- Constants ---
INDEX_DIR = "rag_indexes"
os.makedirs(INDEX_DIR, exist_ok=True) # Ensure the directory for FAISS indexes exists

# --- Helper ---
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

# --- Text Chunking (Unchanged) ---
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- RAG Class ---
class RAGIndex:
    """
    A RAG system that interacts with a SQLite database for text chunks
    and FAISS for vector indexing, on a per-property basis.
    """
    def __init__(self, property_id: int, model_name=EMBED_MODEL):
        """
        Initializes the RAG system for a specific property.

        Args:
            property_id (int): The ID of the property to work with.
            model_name (str): The name of the sentence transformer model.
        """
        if not isinstance(property_id, int):
            raise TypeError("property_id must be an integer.")
            
        self.property_id = property_id
        self.index_path = os.path.join(INDEX_DIR, f"{self.property_id}.index")
        self.model = SentenceTransformer(model_name)
        self.index = None # The FAISS index will be loaded or built on demand

    def build_and_save_index(self, documents: list[tuple[int, str]]):
        """
        Builds a FAISS index from documents, saves it to a file named after the
        property_id, and stores the text chunks in the database.
        
        This will ERASE and REPLACE any existing index and chunks for the property.

        Args:
            documents (list): A list of tuples, where each tuple is
                              (datasource_id, text_content).
        """
        conn = get_db_connection()
        if not conn:
            return

        try:
            with conn: # Use a 'with' statement for automatic transaction handling
                cursor = conn.cursor()
                
                # 1. Clear old chunks for this property's datasources
                # This ensures we don't have stale data from previous indexing runs.
                cursor.execute("""
                    DELETE FROM chunks WHERE datasource_id IN (
                        SELECT id FROM datasources WHERE property_id = ?
                    )
                """, (self.property_id,))
                print(f"Cleared old chunks for property ID {self.property_id}.")

                # 2. Chunk all documents and prepare them for DB insertion
                all_chunks = []
                chunk_to_datasource_map = []
                for datasource_id, text in documents:
                    chunks = chunk_text(text)
                    all_chunks.extend(chunks)
                    chunk_to_datasource_map.extend([datasource_id] * len(chunks))
                
                if not all_chunks:
                    print("No text chunks were generated from the documents. Aborting.")
                    return
                
                # 3. Insert new chunks into the database
                # The chunk ID from the DB will correspond to the vector index
                # This relies on standard SQLite rowid behavior for this table.
                chunk_data_for_db = [
                    (datasource_id, chunk_text) 
                    for datasource_id, chunk_text in zip(chunk_to_datasource_map, all_chunks)
                ]
                cursor.executemany(
                    "INSERT INTO chunks (datasource_id, chunk_text) VALUES (?, ?)",
                    chunk_data_for_db
                )
                print(f"Stored {len(all_chunks)} new chunks in the database.")

                # 4. Create and save the FAISS index
                print("Encoding text chunks... This may take a moment.")
                embeddings = self.model.encode(all_chunks, show_progress_bar=True)
                embeddings = np.array(embeddings).astype('float32')

                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings)
                
                faiss.write_index(self.index, self.index_path)
                print(f"Index built and saved to '{self.index_path}'.")

        except Exception as e:
            print(f"❌ An error occurred during index building: {e}")
        finally:
            conn.close()

    def _load_index(self):
        """Loads the FAISS index from disk if it's not already loaded."""
        if self.index is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Index file not found for property {self.property_id} at {self.index_path}. Please run indexing first.")
            print(f"Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Searches for a query and retrieves the top-k relevant text chunks
        from the database.
        """
        self._load_index() # Ensure the index is loaded

        q_emb = self.model.encode([query]).astype('float32')
        _, I = self.index.search(q_emb, top_k)
        
        chunk_ids = [int(i) for i in I[0]]
        
        # Now, retrieve the text for these chunks from the database
        conn = get_db_connection()
        if not conn:
            return []
            
        try:
            cursor = conn.cursor()
            # The '?' placeholder syntax automatically creates a tuple for the IN clause
            placeholders = ','.join('?' for _ in chunk_ids)
            sql = f"SELECT chunk_text FROM chunks WHERE rowid IN ({placeholders})"
            
            cursor.execute(sql, chunk_ids)
            results = cursor.fetchall()
            return [row['chunk_text'] for row in results]
            
        except Exception as e:
            print(f"❌ An error occurred while retrieving chunks from DB: {e}")
            return []
        finally:
            conn.close()
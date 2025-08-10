# lise/rag.py

import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
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

# --- NEW: Smart Text Chunking Function ---
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
        # This preserves the original formatting.
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
    A RAG system that interacts with a SQLite database for text chunks
    and FAISS for vector indexing, on a per-property basis.
    """
    def __init__(self, property_id: int, model_name=EMBED_MODEL):
        if not isinstance(property_id, int):
            raise TypeError("property_id must be an integer.")
            
        self.property_id = property_id
        self.index_path = os.path.join(INDEX_DIR, f"{self.property_id}.index")
        self.model = SentenceTransformer(model_name)
        self.index = None

    def build_and_save_index(self, documents: list[tuple[int, str]]):
        """
        Builds a FAISS index from documents, using smart chunking, and stores
        the chunks in the database. This will ERASE and REPLACE any existing
        index and chunks for the datasources of this property.
        """
        conn = get_db_connection()
        if not conn: return

        try:
            with conn: # Use a 'with' statement for automatic transaction handling
                cursor = conn.cursor()
                
                # 1. Clear old chunks for this property's datasources
                cursor.execute("""
                    DELETE FROM chunks WHERE datasource_id IN (
                        SELECT id FROM datasources WHERE property_id = ?
                    )
                """, (self.property_id,))
                print(f"-> Cleared old chunks for property ID {self.property_id}.")

                # 2. Chunk all documents using the new smart splitter
                all_chunks = []
                chunk_to_datasource_map = []
                for datasource_id, text in documents:
                    chunks = recursive_character_splitter(text)
                    all_chunks.extend(chunks)
                    chunk_to_datasource_map.extend([datasource_id] * len(chunks))
                
                if not all_chunks:
                    print("-> No text chunks were generated from the documents. Aborting.")
                    return
                
                # 3. Insert new chunks into the database
                chunk_data_for_db = [
                    (datasource_id, chunk_text) 
                    for datasource_id, chunk_text in zip(chunk_to_datasource_map, all_chunks)
                ]
                cursor.executemany(
                    "INSERT INTO chunks (datasource_id, chunk_text) VALUES (?, ?)",
                    chunk_data_for_db
                )
                print(f"-> Stored {len(all_chunks)} new chunks in the database.")

                # 4. Create and save the FAISS index
                print("-> Encoding text chunks for vector index... This may take a moment.")
                embeddings = self.model.encode(all_chunks, show_progress_bar=True)
                embeddings = np.array(embeddings).astype('float32')

                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(embeddings)
                
                faiss.write_index(self.index, self.index_path)
                print(f"-> Index built and saved to '{self.index_path}'.")

        except Exception as e:
            print(f"❌ An error occurred during index building: {e}")
        finally:
            if conn:
                conn.close()

    def _load_index(self):
        """Loads the FAISS index from disk if it's not already loaded."""
        if self.index is None:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"Index file not found for property {self.property_id} at {self.index_path}.")
            self.index = faiss.read_index(self.index_path)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Searches for a query and retrieves the top-k relevant text chunks from the database.
        """
        self._load_index()

        q_emb = self.model.encode([query]).astype('float32')
        _, I = self.index.search(q_emb, top_k)
        
        faiss_indices = [int(i) for i in I[0]]
        
        conn = get_db_connection()
        if not conn: return []
            
        try:
            cursor = conn.cursor()
            
            # Find the minimum rowid for this property's chunks to map FAISS index to DB rowid
            cursor.execute("""
                SELECT MIN(rowid) FROM chunks WHERE datasource_id IN (
                    SELECT id FROM datasources WHERE property_id = ?
                )
            """, (self.property_id,))
            min_rowid_result = cursor.fetchone()

            if not min_rowid_result or min_rowid_result[0] is None:
                return []
            
            base_rowid = min_rowid_result[0]
            
            # Correctly map the 0-based FAISS index to the database's sequential rowid
            db_chunk_ids = [base_rowid + idx for idx in faiss_indices]
            
            placeholders = ','.join('?' for _ in db_chunk_ids)
            sql = f"SELECT chunk_text FROM chunks WHERE rowid IN ({placeholders})"
            
            # We need to preserve the order returned by FAISS
            cursor.execute(sql, db_chunk_ids)
            results_dict = {row['rowid']: row['chunk_text'] for row in cursor.fetchall()}
            
            # Return chunks in the order of relevance found by FAISS
            ordered_chunks = [results_dict[db_id] for db_id in db_chunk_ids if db_id in results_dict]
            
            return ordered_chunks
            
        except Exception as e:
            print(f"❌ An error occurred while retrieving chunks from DB: {e}")
            return []
        finally:
            if conn:
                conn.close()
# rag.py

import os
import json
from lise.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

class RAGIndex:
    """A RAG system that can save/load a persistent index."""
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def build_index(self, documents):
        """Accepts a list of (url, text), splits, embeds, and indexes"""
        all_chunks = []
        for url, text in documents:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        self.text_chunks = all_chunks
        print("Encoding text chunks... This may take a moment.")
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        
        # Ensure embeddings are float32
        embeddings = np.array(embeddings).astype('float32')

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print("Index built successfully.")

    def retrieve(self, query, top_k=5):
        """Returns top-k relevant text chunks from the loaded index."""
        if self.index is None:
            raise RuntimeError("Index is not built or loaded. Please build or load an index first.")
        q_emb = self.model.encode([query]).astype('float32')
        _, I = self.index.search(q_emb, top_k)
        return [self.text_chunks[i] for i in I[0]]

    def save_index(self, index_path, chunks_path):
        """Saves the FAISS index and text chunks to disk."""
        print(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving text chunks to {chunks_path}")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.text_chunks, f, ensure_ascii=False, indent=2)

    def load_index(self, index_path, chunks_path):
        """Loads a FAISS index and text chunks from disk."""
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            raise FileNotFoundError("Index or chunks file not found.")
            
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        
        print(f"Loading text chunks from {chunks_path}")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.text_chunks = json.load(f)
        print("Index and chunks loaded successfully.")
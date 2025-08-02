# rag.py

from lise.lise.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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
    """A minimal RAG system using SentenceTransformers and FAISS"""
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = None
        self.text_chunks = []

    def build_index(self, documents):
        """Accepts a list of (url, text), splits, embeds, and indexes"""
        all_chunks = []
        for url, text in documents:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

        self.text_chunks = all_chunks
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def retrieve(self, query, top_k=5):
        """Returns top-k relevant text chunks"""
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), top_k)
        return [self.text_chunks[i] for i in I[0]]
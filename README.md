# lise – Website Chatbot (GROQ + RAG)

Minimal, command-line based chatbot that crawls a website, builds a semantic index, and responds to user queries using the GROQ API and Retrieval-Augmented Generation.

Named in honor of Odette Marie Léonie Céline Hallowes, GC, MBE (née Brailly; 28 April 1912 – 13 March 1995). Also known as Odette Churchill and Odette Sansom, code named Lise, was an agent for the United Kingdom's clandestine Special Operations Executive (SOE) in France during the Second World War. 

## Features
- Crawls any static website
- RAG with Sentence-Transformers + FAISS
- Fast LLM inference via GROQ (LLaMA 3)
- Lightweight CLI only (no GUI)

## Install

```bash
pip install -r requirements.txt
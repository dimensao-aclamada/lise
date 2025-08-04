# manage.py

import os
import json
import argparse
from urllib.parse import urlparse

# Assuming your project structure allows this import
from lise.crawler import crawl_website
from lise.rag import RAGIndex

CONFIG_FILE = "websites.json"
CHUNKS_DIR = "website_chunks"
INDEX_DIR = "rag_indexes" # New directory for persistent indexes

def load_websites_config():
    """Loads or initializes the websites.json configuration file."""
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_websites_config(config):
    """Saves the websites configuration."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def add_data_source(name: str, url: str):
    """Adds a new data source to the configuration."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    clean_domain = urlparse(url).netloc
    
    config = load_websites_config()
    if name in config:
        print(f"Error: Data source '{name}' already exists.")
        return

    config[name] = {
        "website": clean_domain,
        "mandatory_pages": ["/"],
        "exclude_pages": [],
        "instructions": "Default crawl. No specific instruction."
    }
    save_websites_config(config)
    print(f"Data source '{name}' for website '{clean_domain}' added to {CONFIG_FILE}.")

def crawl_and_index(name: str):
    """Crawls a website and builds a persistent RAG index."""
    config = load_websites_config()
    if name not in config:
        print(f"Error: Data source '{name}' not found. Please add it first using the 'add' command.")
        return

    source_config = config[name]
    base_url = "https://" + source_config["website"]

    print(f"Starting crawl for '{name}' ({base_url})...")
    pages = crawl_website(
        base_url=base_url,
        must_include=source_config.get("mandatory_pages", []),
        must_exclude=source_config.get("exclude_pages", [])
    )

    if not pages:
        print("Error: Crawling failed, no pages were retrieved. Aborting.")
        return

    print(f"Crawled {len(pages)} pages successfully.")

    # Build the RAG index
    rag = RAGIndex()
    rag.build_index(pages)

    # Define paths for saving
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, f"{name}.index")
    chunks_path = os.path.join(INDEX_DIR, f"{name}_chunks.json")

    # Save the index and chunks
    rag.save_index(index_path, chunks_path)
    print(f"Index for '{name}' is built and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage chatbot data sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add command
    parser_add = subparsers.add_parser("add", help="Add a new data source.")
    parser_add.add_argument("name", type=str, help="A unique name for the data source (e.g., 'mysite').")
    parser_add.add_argument("url", type=str, help="The full base URL of the website (e.g., 'https://example.com').")

    # Crawl and Index command
    parser_crawl = subparsers.add_parser("crawl", help="Crawl a website and build its index.")
    parser_crawl.add_argument("name", type=str, help="The name of the data source to crawl.")

    args = parser.parse_args()

    if args.command == "add":
        add_data_source(args.name, args.url)
    elif args.command == "crawl":
        crawl_and_index(args.name)
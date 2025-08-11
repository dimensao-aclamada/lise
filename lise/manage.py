# lise/manage.py

import argparse
import secrets
import sqlite3
import os
import json
import requests # <-- ADDED: Needed for downloading from URLs
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

# --- Lise Imports ---
from lise.config import DATABASE_FILE
from lise.encryption import encrypt_key
from lise.crawler import crawl_website
from lise.rag import RAGIndex, recursive_character_splitter

# Load environment variables at the top
load_dotenv()

# --- Constants ---
CHUNK_FILES_DIR = "chunk_files"
os.makedirs(CHUNK_FILES_DIR, exist_ok=True)

# --- Database Helper (Unchanged) ---
def get_db_connection():
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        exit(1)

# --- Property Management (Unchanged) ---
def create_property(name: str, website: str, platform_name: str, platform_api_key: str = None):
    if platform_api_key is None:
        if platform_name == 'groq':
            platform_api_key = os.getenv("GROQ_API_KEY")
            if platform_api_key: print("Read GROQ_API_KEY from .env file.")
    if not platform_api_key:
        print(f"❌ Error: Platform API key for '{platform_name}' not found.")
        return
    conn = get_db_connection()
    try:
        parsed_url = urlparse(website);
        if not parsed_url.scheme: parsed_url = parsed_url._replace(scheme="https")
        normalized_website = urlunparse(parsed_url._replace(path="", params="", query="", fragment=""))
        lise_api_key = f"lise_{secrets.token_hex(24)}"
        encrypted_platform_key = encrypt_key(platform_api_key)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO properties (name, website, lise_api_key, platform_name, platform_api_key) VALUES (?, ?, ?, ?, ?)",(name, normalized_website, lise_api_key, platform_name, encrypted_platform_key))
        property_id = cursor.lastrowid
        cursor.execute("INSERT INTO datasources (property_id, type, source_uri) VALUES (?, 'website', ?)",(property_id, normalized_website))
        conn.commit()
        print("-" * 60); print(f"✅ Property '{name}' created successfully for website {normalized_website}."); print("\n   This is the only time your Lise API key will be displayed."); print("   Please save it securely."); print(f"\n   >> Lise API Key: {lise_api_key} <<"); print("-" * 60)
    except sqlite3.IntegrityError: print(f"❌ Error: A property with the name '{name}' or website '{website}' already exists.")
    except Exception as e: conn.rollback(); print(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn: conn.close()

def list_properties():
    conn = get_db_connection();
    try:
        cursor = conn.cursor(); cursor.execute("SELECT id, name, website FROM properties ORDER BY created_at DESC"); properties = cursor.fetchall()
        if not properties: print("No properties found."); return
        print(f"{'ID':<5} {'Name':<30} {'Website'}"); print("-" * 80)
        for prop in properties: print(f"{prop['id']:<5} {prop['name']:<30} {prop['website']}")
    except Exception as e: print(f"❌ An error occurred: {e}")
    finally:
        if conn: conn.close()

def delete_property(property_id: int):
    conn = get_db_connection();
    try:
        cursor = conn.cursor(); cursor.execute("SELECT name FROM properties WHERE id = ?", (property_id,)); prop = cursor.fetchone()
        if not prop: print(f"❌ Error: Property with ID '{property_id}' not found."); return
        confirm = input(f"❓ Are you sure you want to delete property '{prop['name']}' (ID: {property_id})? [y/N]: ")
        if confirm.lower() != 'y': print("Aborted."); return
        with conn:
            cursor.execute("DELETE FROM properties WHERE id = ?", (property_id,))
            index_file = os.path.join("rag_indexes", f"{property_id}.index");
            if os.path.exists(index_file): os.remove(index_file); print(f"-> Deleted index file: {index_file}")
        print(f"✅ Property '{prop['name']}' and its associated data have been deleted.")
    except Exception as e: print(f"❌ An error occurred: {e}")
    finally:
        if conn: conn.close()

# --- Indexing Workflow ---

def generate_chunks_file(website: str):
    print(f"\n--- Generating chunks for website: {website} ---")
    try:
        pages = crawl_website(website)
        if not pages:
            print("Crawling failed, no pages retrieved. Aborting."); return
        all_chunks = []
        for _url, text in pages:
            all_chunks.extend(recursive_character_splitter(text))
        if not all_chunks:
            print("No text chunks were generated from the crawled content."); return
        sanitized_filename = urlparse(website).netloc.replace('.', '_') + "_chunks.json"
        output_path = os.path.join(CHUNK_FILES_DIR, sanitized_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Successfully generated {len(all_chunks)} chunks."); print("   File saved to:"); print(f"   >> {output_path} <<")
        print("\nNext, you can inspect this file and then run the 'datasources:load' command.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during chunk generation: {e}")

# --- THIS IS THE FULLY CORRECTED FUNCTION ---
def load_chunks_and_index(website: str, chunks_source: str):
    """
    Loads chunks from a local file OR a remote URL, builds the FAISS index,
    and updates the corresponding property in the database with the source URL.
    """
    print(f"\n--- Loading chunks from '{chunks_source}' for property: {website} ---")

    # 1. Load chunks from the source (either URL or local file)
    chunks = None
    try:
        if chunks_source.startswith(('http://', 'https://')):
            print("-> Source is a remote URL. Downloading...")
            response = requests.get(chunks_source, timeout=10)
            response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
            chunks = response.json()
        else: # Assumed to be a local file path
            if not os.path.exists(chunks_source):
                print(f"❌ Error: Local file not found at '{chunks_source}'.")
                return
            with open(chunks_source, "r", encoding="utf-8") as f:
                chunks = json.load(f)
    except Exception as e:
        print(f"❌ Error loading or parsing chunks from source: {e}")
        return

    # 2. Validate chunks and find the property in the database
    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        print("❌ Error: The provided source is not a valid JSON array of strings."); return
    
    print(f"Loaded {len(chunks)} chunks successfully.")

    conn = get_db_connection()
    try:
        parsed_url = urlparse(website);
        if not parsed_url.scheme: parsed_url = parsed_url._replace(scheme="https")
        normalized_website = urlunparse(parsed_url._replace(path="", params="", query="", fragment=""))
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM properties WHERE website = ?", (normalized_website,))
        prop_row = cursor.fetchone()
        if not prop_row:
            print(f"❌ Error: No property found for website '{normalized_website}'."); return
        
        property_id = prop_row['id']
        property_name = prop_row['name']
        print(f"Found property '{property_name}' (ID: {property_id}).")

        # 3. Build the FAISS index from the loaded chunks
        rag = RAGIndex(property_id=property_id)
        rag.build_index_from_chunks(chunks)

        # 4. Determine the URL to save in the database
        if chunks_source.startswith(('http://', 'https://')):
            db_url_to_store = chunks_source
        else:
            db_url_to_store = f"file://{os.path.abspath(chunks_source)}"

        # 5. Update the datasource in the database with the URL
        with conn:
            cursor.execute("SELECT id FROM datasources WHERE property_id = ? AND type = 'website' LIMIT 1", (property_id,))
            ds_row = cursor.fetchone()
            if not ds_row:
                print("❌ Error: Could not find a 'website' type datasource for this property."); return
            
            datasource_id = ds_row['id']
            current_time = sqlite3.datetime.datetime.now()
            
            conn.execute(
                "UPDATE datasources SET status = 'completed', chunks_json_url = ?, index_updated_at = ? WHERE id = ?",
                (db_url_to_store, current_time, datasource_id)
            )

        print(f"\n✅ Indexing complete for property '{property_name}'.")
        print(f"   Database updated with chunk URL: {db_url_to_store}")

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn: conn.close()


# --- Main Argparse CLI (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lise administration tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_create = subparsers.add_parser("properties:create", help="Create a new property.")
    p_create.add_argument("name"); p_create.add_argument("--website", required=True)
    p_create.add_argument("--platform-name", choices=["groq", "openai"], default="groq"); p_create.add_argument("--platform-key")
    p_list = subparsers.add_parser("properties:list", help="List all properties.")
    p_delete = subparsers.add_parser("properties:delete", help="Delete a property."); p_delete.add_argument("property_id", type=int)

    ds_generate = subparsers.add_parser("datasources:generate", help="STEP 1: Crawl website and generate a chunks.json file.")
    ds_generate.add_argument("--website", required=True)
    
    ds_load = subparsers.add_parser("datasources:load", help="STEP 2: Load chunks file/URL, build index, and update DB.")
    ds_load.add_argument("--website", required=True)
    ds_load.add_argument("--chunks-file", required=True, help="Path or URL to the chunks.json file.")
    
    args = parser.parse_args()

    if args.command == "properties:create":
        create_property(args.name, args.website, args.platform_name, args.platform_key)
    elif args.command == "properties:list":
        list_properties()
    elif args.command == "properties:delete":
        delete_property(args.property_id)
    elif args.command == "datasources:generate":
        generate_chunks_file(args.website)
    elif args.command == "datasources:load":
        load_chunks_and_index(args.website, args.chunks_file)
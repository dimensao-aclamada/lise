# lise/manage.py

import argparse
import secrets
import sqlite3
import os
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv

# --- Lise Imports ---
from lise.config import DATABASE_FILE
from lise.encryption import encrypt_key
from lise.crawler import crawl_website
from lise.rag import RAGIndex

# Load environment variables from .env file at the very top
load_dotenv()

# --- Database Helper ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        exit(1)

# --- Property Management ---
def create_property(name: str, website: str, platform_name: str, platform_api_key: str = None):
    """
    Creates a new property, gets the platform key from the command line or .env,
    and adds its website as the first datasource.
    """
    # If the platform key was not provided via the command line, get it from the environment
    if platform_api_key is None:
        if platform_name == 'groq':
            platform_api_key = os.getenv("GROQ_API_KEY")
            if platform_api_key:
                print("Read GROQ_API_KEY from .env file.")
        # Add other platforms here if needed, e.g., elif platform_name == 'openai': ...

    if not platform_api_key:
        print(f"❌ Error: Platform API key for '{platform_name}' not found. Please provide it via the --platform-key argument or set it as GROQ_API_KEY in your .env file.")
        return

    conn = get_db_connection()
    try:
        # Normalize the website URL to a standard format (e.g., https://example.com)
        parsed_url = urlparse(website)
        if not parsed_url.scheme:
            parsed_url = parsed_url._replace(scheme="https")
        normalized_website = urlunparse(parsed_url._replace(path="", params="", query="", fragment=""))
        
        lise_api_key = f"lise_{secrets.token_hex(24)}"
        encrypted_platform_key = encrypt_key(platform_api_key)

        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO properties (name, website, lise_api_key, platform_name, platform_api_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, normalized_website, lise_api_key, platform_name, encrypted_platform_key)
        )
        property_id = cursor.lastrowid
        
        cursor.execute(
            """
            INSERT INTO datasources (property_id, type, source_uri)
            VALUES (?, 'website', ?)
            """,
            (property_id, normalized_website)
        )
        
        conn.commit()
        
        print("-" * 60)
        print(f"✅ Property '{name}' created successfully for website {normalized_website}.")
        print("\n   This is the only time your Lise API key will be displayed.")
        print("   Please save it securely.")
        print(f"\n   >> Lise API Key: {lise_api_key} <<")
        print("-" * 60)

    except sqlite3.IntegrityError:
        print(f"❌ Error: A property with the name '{name}' or website '{website}' already exists.")
    except Exception as e:
        conn.rollback()
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

def list_properties():
    """Lists all properties in the database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, website FROM properties ORDER BY created_at DESC")
        properties = cursor.fetchall()
        
        if not properties:
            print("No properties found. Use 'properties:create' to add one.")
            return
            
        print(f"{'ID':<5} {'Name':<30} {'Website'}")
        print("-" * 80)
        for prop in properties:
            print(f"{prop['id']:<5} {prop['name']:<30} {prop['website']}")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if conn:
            conn.close()

def delete_property(property_id: int):
    """Deletes a property and all its associated datasources and chunks from the database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM properties WHERE id = ?", (property_id,))
        prop = cursor.fetchone()
        if not prop:
            print(f"❌ Error: Property with ID '{property_id}' not found.")
            return
        
        confirm = input(f"❓ Are you sure you want to delete property '{prop['name']}' (ID: {property_id})? This will also delete its index file and cannot be undone. [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

        with conn:
            # The 'ON DELETE CASCADE' in our schema will handle deleting datasources and chunks automatically.
            cursor.execute("DELETE FROM properties WHERE id = ?", (property_id,))
            
            # Clean up the associated index file
            index_file = os.path.join("rag_indexes", f"{property_id}.index")
            if os.path.exists(index_file):
                os.remove(index_file)
                print(f"-> Deleted index file: {index_file}")

        print(f"✅ Property '{prop['name']}' and its associated data have been deleted.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if conn:
            conn.close()

# --- Datasource Management ---
def index_datasources_by_website(website: str):
    """
    Finds a property by its website URL and indexes all its datasources.
    """
    print(f"\n--- Starting indexing process for property associated with: {website} ---")
    conn = get_db_connection()
    try:
        parsed_url = urlparse(website)
        if not parsed_url.scheme:
            parsed_url = parsed_url._replace(scheme="httpshttps")
        normalized_website = urlunparse(parsed_url._replace(path="", params="", query="", fragment=""))

        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM properties WHERE website = ?", (normalized_website,))
        prop_row = cursor.fetchone()
        
        if not prop_row:
            print(f"❌ Error: No property found with the website '{normalized_website}'.")
            return

        property_id = prop_row['id']
        property_name = prop_row['name']
        print(f"Property found: '{property_name}' (ID: {property_id})")

        cursor.execute("SELECT id, type, source_uri FROM datasources WHERE property_id = ?", (property_id,))
        datasources_to_index = cursor.fetchall()
        
        if not datasources_to_index:
            print(f"No datasources found for property '{property_name}'. Nothing to index.")
            return

        all_documents = []
        for ds in datasources_to_index:
            print(f"-> Processing datasource ID {ds['id']} ({ds['type']}): {ds['source_uri']}")
            if ds['type'] == 'website':
                pages = crawl_website(ds['source_uri'])
                for _url, text in pages:
                    all_documents.append((ds['id'], text))
                print(f"   Crawled {len(pages)} pages.")
            else:
                print(f"   Skipping. Indexing for type '{ds['type']}' is not yet implemented.")

        if not all_documents:
            print("No content could be retrieved from any datasource. Aborting index process.")
            return
        
        rag = RAGIndex(property_id=property_id)
        rag.build_and_save_index(all_documents)
        
        with conn:
            datasource_ids = [ds['id'] for ds in datasources_to_index]
            current_time = sqlite3.datetime.datetime.now()
            update_data = [('completed', current_time, ds_id) for ds_id in datasource_ids]
            conn.executemany("UPDATE datasources SET status = ?, last_indexed_at = ? WHERE id = ?", update_data)

        print(f"\n✅ Indexing process complete for property '{property_name}'.")

    except Exception as e:
        print(f"❌ An unexpected error occurred during indexing: {e}")
    finally:
        if conn:
            conn.close()

# --- Main Argparse CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lise administration tool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Properties Commands ---
    p_create = subparsers.add_parser("properties:create", help="Create a new property.")
    p_create.add_argument("name", help="A unique name for the property.")
    p_create.add_argument("--website", help="The primary website URL for the property.", required=True)
    p_create.add_argument("--platform-name", help="The LLM platform to use.", choices=["groq", "openai"], default="groq")
    p_create.add_argument("--platform-key", help="[Optional] The API key for the LLM platform. If not provided, it's read from GROQ_API_KEY env var.")

    p_list = subparsers.add_parser("properties:list", help="List all existing properties.")
    
    p_delete = subparsers.add_parser("properties:delete", help="Delete a property and all its associated data.")
    p_delete.add_argument("property_id", type=int, help="The ID of the property to delete.")

    # --- Datasources Commands ---
    ds_index = subparsers.add_parser("datasources:index", help="Crawl and index all datasources for a property, identified by its website.")
    ds_index.add_argument("--website", type=str, help="The primary website URL of the property to index.", required=True)
    
    args = parser.parse_args()

    if args.command == "properties:create":
        create_property(args.name, args.website, args.platform_name, args.platform_key)
    elif args.command == "properties:list":
        list_properties()
    elif args.command == "properties:delete":
        delete_property(args.property_id)
    elif args.command == "datasources:index":
        index_datasources_by_website(args.website)
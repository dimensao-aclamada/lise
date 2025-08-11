# database_setup.py
import sqlite3
import os

DATABASE_FILE = "lise.db"

# SQL for the 'properties' table (Unchanged from the previous version)
SQL_CREATE_PROPERTIES_TABLE = """
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    lise_api_key TEXT NOT NULL UNIQUE,
    website TEXT UNIQUE,
    platform_name TEXT NOT NULL,
    platform_api_key BLOB NOT NULL,
    created_at DATETIME NOT NULL DEFAULT (datetime('now'))
);
"""

# SQL for the 'datasources' table is MODIFIED for the Remote Chunks architecture
SQL_CREATE_DATASOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS datasources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    chunks_json_url TEXT, -- <-- ADDED: Will store the URL to the chunks file.
    index_updated_at DATETIME, -- Renamed from last_indexed_at for clarity
    created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (property_id) REFERENCES properties (id) ON DELETE CASCADE
);
"""

def setup_database():
    """
    Connects to the SQLite database and sets up the tables for the
    'Remote Chunks' architecture. This involves creating 'properties' and
    'datasources' tables and ensuring the 'chunks' table does not exist.
    """
    conn = None
    try:
        print(f"Connecting to database: '{DATABASE_FILE}'...")
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        print("Connection successful.")

        # It's a good practice to explicitly enable foreign key support
        print("Enabling foreign key support...")
        cursor.execute("PRAGMA foreign_keys = ON;")

        print("Creating 'properties' table (if not exists)...")
        cursor.execute(SQL_CREATE_PROPERTIES_TABLE)

        print("Creating 'datasources' table with 'chunks_json_url' column (if not exists)...")
        cursor.execute(SQL_CREATE_DATASOURCES_TABLE)
        
        # We explicitly REMOVE the chunks table if it exists from the old schema
        # This makes the script safely migratable from the previous architecture.
        print("Dropping old 'chunks' table (if it exists) to finalize migration...")
        cursor.execute("DROP TABLE IF EXISTS chunks;")

        # Commit all changes to the database
        conn.commit()
        
        print("\nDatabase setup complete for the 'Remote Chunks' architecture.")
        print(f"The database file '{DATABASE_FILE}' is ready.")

    except sqlite3.Error as e:
        print(f"An error occurred during database setup: {e}")

    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    # This block allows the script to be run directly
    if os.path.exists(DATABASE_FILE):
        print(f"Database file '{DATABASE_FILE}' already exists. Schema will be updated if necessary.")
    setup_database()
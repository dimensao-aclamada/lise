# database_setup.py
import sqlite3
import os

# --- Configuration ---
DATABASE_FILE = "lise.db"

# --- SQL Statements for Table Creation ---

# SQL for the 'properties' table
# This table holds the primary client/project information.
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

# SQL for the 'datasources' table
# This table stores all indexable content associated with a property.
SQL_CREATE_DATASOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS datasources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    property_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    last_indexed_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (property_id) REFERENCES properties (id) ON DELETE CASCADE
);
"""

SQL_CREATE_CHUNKS_TABLE = """CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    datasource_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    FOREIGN KEY (datasource_id) REFERENCES datasources (id) ON DELETE CASCADE
);"""

def setup_database():
    """
    Connects to the SQLite database, creates the necessary tables
    if they don't already exist, and ensures foreign key support is enabled.
    """
    conn = None
    try:
        print(f"Connecting to database: '{DATABASE_FILE}'...")
        # The connect() function will create the database file if it doesn't exist.
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        print("Connection successful.")

        print("Creating 'properties' table...")
        cursor.execute(SQL_CREATE_PROPERTIES_TABLE)

        print("Creating 'datasources' table...")
        cursor.execute(SQL_CREATE_DATASOURCES_TABLE)
        
        print("Creating 'chunks' table...")
        cursor.execute(SQL_CREATE_CHUNKS_TABLE)
        
        # It's a good practice to explicitly enable foreign key constraints for each connection.
        print("Enabling foreign key support...")
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Commit the changes to the database
        conn.commit()
        
        print("\nDatabase setup complete.")
        print(f"The database file '{DATABASE_FILE}' is ready.")

    except sqlite3.Error as e:
        print(f"An error occurred during database setup: {e}")

    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    # This block runs only when the script is executed directly
    if os.path.exists(DATABASE_FILE):
        print(f"Database file '{DATABASE_FILE}' already exists.")
        # We can still run setup() because of "IF NOT EXISTS",
        # which makes the script safe to run multiple times.
    setup_database()
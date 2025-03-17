# create_api_keys_table.py
import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    print("❌ ERROR: DATABASE_URL environment variable not set.")
    exit(1)

try:
    # Connect to the database
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    
    # Create the api_keys table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_keys (
        id SERIAL PRIMARY KEY,
        key_hash TEXT NOT NULL UNIQUE,
        client_name TEXT NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        permissions TEXT
    )
    """)
    
    # Create index for faster lookups
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)
    """)
    
    # Commit the changes
    conn.commit()
    print("✅ API Keys table created successfully!")
    
except Exception as e:
    print(f"❌ Error creating API Keys table: {e}")
finally:
    if 'conn' in locals():
        cursor.close()
        conn.close()
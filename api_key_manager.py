# api_key_manager.py
import os
import secrets
import hashlib
import time
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def generate_api_key(length=32):
    """Generate a secure random API key"""
    # Use secrets module for cryptographically strong random values
    return secrets.token_hex(length)

def hash_api_key(api_key):
    """Hash an API key using SHA-256"""
    return hashlib.sha256(api_key.encode()).hexdigest()

def create_api_key(client_name, expires_in_days=None):
    """
    Create a new API key for a client
    
    Args:
        client_name: Name of the client/partner
        expires_in_days: Number of days until key expires (None = never expires)
        
    Returns:
        dict: Information about the created key including the raw key (only returned once)
    """
    # Generate a new API key
    api_key = generate_api_key()
    
    # Hash the key for storage
    key_hash = hash_api_key(api_key)
    
    # Calculate expiration date if provided
    expires_at = None
    if expires_in_days is not None:
        expires_at = datetime.now() + timedelta(days=expires_in_days)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Insert the new key into the database
        cursor.execute(
            """
            INSERT INTO api_keys (key_hash, client_name, is_active, expires_at)
            VALUES (%s, %s, %s, %s)
            RETURNING id, client_name, is_active, created_at, expires_at
            """,
            (key_hash, client_name, True, expires_at)
        )
        
        # Get the inserted record
        key_info = cursor.fetchone()
        conn.commit()
        
        # Add the raw key to the returned info (this is the only time it will be available)
        key_info['api_key'] = api_key
        
        return key_info
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def get_api_keys():
    """Get all API keys (without the raw keys)"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute(
            """
            SELECT id, client_name, is_active, created_at, expires_at
            FROM api_keys
            ORDER BY created_at DESC
            """
        )
        
        keys = cursor.fetchall()
        
        # Convert datetime objects to strings
        for key in keys:
            if key['created_at']:
                key['created_at'] = key['created_at'].isoformat()
            if key['expires_at']:
                key['expires_at'] = key['expires_at'].isoformat()
                
        return keys
        
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()

def revoke_api_key(key_id):
    """Revoke an API key by setting is_active to False"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            UPDATE api_keys
            SET is_active = FALSE
            WHERE id = %s
            RETURNING id
            """,
            (key_id,)
        )
        
        result = cursor.fetchone()
        conn.commit()
        
        return result is not None
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

def validate_api_key(api_key):
    """
    Validate an API key
    
    Args:
        api_key: The API key to validate
        
    Returns:
        dict: Key information if valid, None if invalid
    """
    if not api_key:
        return None
    
    # Hash the provided key
    key_hash = hash_api_key(api_key)
    
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Look up the key hash
        cursor.execute(
            """
            SELECT id, client_name, is_active, created_at, expires_at
            FROM api_keys
            WHERE key_hash = %s
            """,
            (key_hash,)
        )
        
        key_info = cursor.fetchone()
        
        # Key not found
        if not key_info:
            return None
            
        # Key is not active
        if not key_info['is_active']:
            return None
            
        # Key is expired
        if key_info['expires_at'] and key_info['expires_at'] < datetime.now():
            return None
            
        return key_info
        
    except Exception as e:
        raise e
    finally:
        cursor.close()
        conn.close()

# Command-line interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python api_key_manager.py [create|list|revoke|validate]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    try:
        if command == "create":
            if len(sys.argv) < 3:
                print("Usage: python api_key_manager.py create <client_name> [expires_in_days]")
                sys.exit(1)
                
            client_name = sys.argv[2]
            expires_in_days = None
            
            if len(sys.argv) >= 4:
                expires_in_days = int(sys.argv[3])
                
            key_info = create_api_key(client_name, expires_in_days)
            print(f"✅ Created API key for {client_name}:")
            print(f"API Key: {key_info['api_key']} (SAVE THIS - IT WON'T BE SHOWN AGAIN)")
            print(f"ID: {key_info['id']}")
            print(f"Created: {key_info['created_at']}")
            if key_info['expires_at']:
                print(f"Expires: {key_info['expires_at']}")
                
        elif command == "list":
            keys = get_api_keys()
            print(f"📋 Found {len(keys)} API keys:")
            
            for key in keys:
                status = "✅ ACTIVE" if key['is_active'] else "❌ REVOKED"
                expires = f"expires {key['expires_at']}" if key['expires_at'] else "never expires"
                print(f"ID {key['id']} - {key['client_name']} - {status} - {expires}")
                
        elif command == "revoke":
            if len(sys.argv) < 3:
                print("Usage: python api_key_manager.py revoke <key_id>")
                sys.exit(1)
                
            key_id = int(sys.argv[2])
            success = revoke_api_key(key_id)
            
            if success:
                print(f"✅ API key {key_id} revoked successfully")
            else:
                print(f"❌ Failed to revoke API key {key_id} - key not found")
                
        elif command == "validate":
            if len(sys.argv) < 3:
                print("Usage: python api_key_manager.py validate <api_key>")
                sys.exit(1)
                
            api_key = sys.argv[2]
            key_info = validate_api_key(api_key)
            
            if key_info:
                print(f"✅ API key is valid for client: {key_info['client_name']}")
            else:
                print("❌ API key is invalid or revoked")
                
        else:
            print(f"Unknown command: {command}")
            print("Usage: python api_key_manager.py [create|list|revoke|validate]")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
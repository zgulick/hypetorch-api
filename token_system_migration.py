# token_system_migration.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

def run_token_migration():
    """Run the migration to add token system tables and fields."""
    try:
        # Connect to the database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get the correct environment/schema
        db_env = os.environ.get("DB_ENVIRONMENT", "development")
        
        # Set the search path
        cursor.execute(f"SET search_path TO {db_env}")
        
        print("Starting token system migration...")
        
        # 1. Add token fields to api_keys table
        cursor.execute("""
            -- Check if the columns already exist
            DO $$
            BEGIN
                -- Check for token_balance column
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name='api_keys' AND column_name='token_balance'
                ) THEN
                    ALTER TABLE api_keys ADD COLUMN token_balance INTEGER NOT NULL DEFAULT 0;
                    ALTER TABLE api_keys ADD COLUMN tokens_purchased INTEGER NOT NULL DEFAULT 0;
                    ALTER TABLE api_keys ADD COLUMN token_plan TEXT;
                END IF;
            END $$;
        """)
        
        # 2. Create token_transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_transactions (
                id SERIAL PRIMARY KEY,
                api_key_id INTEGER NOT NULL REFERENCES api_keys(id),
                amount INTEGER NOT NULL,
                transaction_type TEXT NOT NULL, -- 'purchase', 'usage', 'refund', etc.
                endpoint TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                request_id TEXT,
                client_ip TEXT,
                metadata JSONB DEFAULT '{}'::jsonb
            );
            
            -- Add indexes for efficient queries
            CREATE INDEX IF NOT EXISTS idx_token_transactions_api_key_id ON token_transactions(api_key_id);
            CREATE INDEX IF NOT EXISTS idx_token_transactions_created_at ON token_transactions(created_at);
            CREATE INDEX IF NOT EXISTS idx_token_transactions_type ON token_transactions(transaction_type);
        """)
        
        # 3. Create a function to check and update token balance
        cursor.execute("""
            CREATE OR REPLACE FUNCTION check_token_balance(key_id INTEGER, required_tokens INTEGER)
            RETURNS BOOLEAN AS $$
            DECLARE
                current_balance INTEGER;
            BEGIN
                -- Get current balance
                SELECT token_balance INTO current_balance FROM api_keys WHERE id = key_id;
                
                -- Check if balance is sufficient
                IF current_balance >= required_tokens THEN
                    -- Update balance
                    UPDATE api_keys SET token_balance = token_balance - required_tokens WHERE id = key_id;
                    RETURN TRUE;
                ELSE
                    RETURN FALSE;
                END IF;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        # 4. Create token purchase function
        cursor.execute("""
            CREATE OR REPLACE FUNCTION add_tokens(key_id INTEGER, amount INTEGER, description TEXT DEFAULT 'Token purchase')
            RETURNS VOID AS $$
            BEGIN
                -- Update token balance
                UPDATE api_keys 
                SET token_balance = token_balance + amount,
                    tokens_purchased = tokens_purchased + amount
                WHERE id = key_id;
                
                -- Record transaction
                INSERT INTO token_transactions 
                    (api_key_id, amount, transaction_type, description)
                VALUES 
                    (key_id, amount, 'purchase', description);
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        # Commit the transaction
        conn.commit()
        print("Token system migration completed successfully!")
        
        return True
    except Exception as e:
        print(f"Error in token system migration: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_token_migration()
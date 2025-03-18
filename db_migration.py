import os
import psycopg2

# Database connection parameters from Render
DATABASE_URL = os.environ.get("DATABASE_URL")

def run_migration():
    try:
        # Establish connection
        conn = psycopg2.connect(DATABASE_URL)
        
        # Create a cursor object
        cur = conn.cursor()
        
        # Migration SQL
        migration_sql = """
        -- Migration script to safely rename an entity

        -- Step 1: Update entity_history table references
        UPDATE entity_history 
        SET entity_name = 'Marina Mabrey' 
        WHERE entity_name = 'Marena Mabry';

        -- Step 2: Update the entities table
        UPDATE entities 
        SET name = 'Marina Mabrey' 
        WHERE name = 'Marena Mabry';
        """
        
        # Execute the migration
        cur.execute(migration_sql)
        
        # Commit the changes
        conn.commit()
        
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        
    finally:
        # Close the connection
        if conn:
            cur.close()
            conn.close()

# Run the migration
if __name__ == "__main__":
    run_migration()
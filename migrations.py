# migrations.py
import os
import logging
import time
from datetime import datetime
from db_pool import DatabaseConnection, execute_query

# Set up a log file specifically for migrations
migration_logger = logging.getLogger('migrations')
file_handler = logging.FileHandler('migrations.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
migration_logger.addHandler(file_handler)
migration_logger.setLevel(logging.INFO)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('migrations')

# Table to track migrations
MIGRATIONS_TABLE = "schema_migrations"

def check_migrations_table():
    """Create the migrations tracking table if it doesn't exist."""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # Check if we're using SQLite
            from db_pool import SQLITE_AVAILABLE, POSTGRESQL_AVAILABLE
            using_sqlite = (POSTGRESQL_AVAILABLE is False or POSTGRESQL_AVAILABLE == "SQLITE")
            
            if using_sqlite:
                # SQLite approach
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{MIGRATIONS_TABLE}'")
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    logger.info(f"Creating migrations table {MIGRATIONS_TABLE} (SQLite)")
                    migration_logger.info(f"Creating migrations table {MIGRATIONS_TABLE} (SQLite)")
                    cursor.execute(f"""
                        CREATE TABLE {MIGRATIONS_TABLE} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            version TEXT NOT NULL,
                            description TEXT,
                            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info("Migrations table created successfully")
                    migration_logger.info("Migrations table created successfully")
                else:
                    logger.info("Migrations table already exists")
                    migration_logger.info("Migrations table already exists")
            else:
                # PostgreSQL approach
                # Get DB_ENVIRONMENT from config
                db_env = os.environ.get("DB_ENVIRONMENT", "development")
                
                # Create schema if it doesn't exist
                cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {db_env}")
                
                # Set search path
                cursor.execute(f"SET search_path TO {db_env}")
                
                # Check if migrations table exists
                cursor.execute(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = '{db_env}'
                        AND table_name = '{MIGRATIONS_TABLE}'
                    )
                """)
                
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    logger.info(f"Creating migrations table {MIGRATIONS_TABLE}")
                    migration_logger.info(f"Creating migrations table {MIGRATIONS_TABLE}")
                    cursor.execute(f"""
                        CREATE TABLE {MIGRATIONS_TABLE} (
                            id SERIAL PRIMARY KEY,
                            version VARCHAR(100) NOT NULL,
                            description TEXT,
                            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
                    logger.info("Migrations table created successfully")
                    migration_logger.info("Migrations table created successfully") 
                else:
                    logger.info("Migrations table already exists")
                    migration_logger.info("Migrations table already exists") 
                
            return True
    except Exception as e:
        logger.error(f"Error checking migrations table: {e}")
        migration_logger.error(f"Error checking migrations table: {e}""Same message")
        return False

def get_applied_migrations():
    """Get list of already applied migrations."""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # For both SQLite and PostgreSQL, this query should work
            cursor.execute(f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY id")
            
            # Handle different result formats
            results = cursor.fetchall()
            
            # Different database drivers might return results differently
            if results and isinstance(results[0], (list, tuple)):
                return [row[0] for row in results]  # Tuple-like results
            elif results and hasattr(results[0], 'get'):
                return [row.get('version') for row in results]  # Dict-like results
            elif results and hasattr(results[0], 'version'):
                return [row.version for row in results]  # Object-like results
            else:
                return []
                
    except Exception as e:
        logger.error(f"Error getting applied migrations: {e}")
        migration_logger.error(f"Error getting applied migrations: {e}")
        return []
    
def record_migration(version, description):
    """Record a successfully applied migration."""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                f"INSERT INTO {MIGRATIONS_TABLE} (version, description) VALUES (%s, %s)", 
                (version, description)
            )
            
            conn.commit()
            logger.info(f"Recorded migration {version}")
            migration_logger.info(f"Recorded migration {version}")
            return True
    except Exception as e:
        logger.error(f"Error recording migration {version}: {e}")
        migration_logger.error(f"Error recording migration {version}: {e}")
        return False

# Define all migrations
# Define all migrations
MIGRATIONS = [
    {
        "version": "1.0.0",
        "description": "Add category field to entities",
        "sql": """
            -- For PostgreSQL
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name='entities' AND column_name='category'
                ) THEN
                    ALTER TABLE entities ADD COLUMN category TEXT DEFAULT 'Sports';
                END IF;
            EXCEPTION WHEN others THEN
                -- This will run if we're on SQLite
                ALTER TABLE entities ADD COLUMN category TEXT DEFAULT 'Sports';
            END $$;
        """,
        "sqlite_sql": """
            -- Check if column exists in SQLite
            PRAGMA table_info(entities);
            -- Add column if it doesn't exist (we'll handle this in Python code)
        """
    },
    {
        "version": "1.0.1",
        "description": "Add subcategory field to entities",
        "sql": """
            -- For PostgreSQL
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name='entities' AND column_name='subcategory'
                ) THEN
                    ALTER TABLE entities ADD COLUMN subcategory TEXT DEFAULT 'Unrivaled';
                END IF;
            EXCEPTION WHEN others THEN
                -- This will run if we're on SQLite
                ALTER TABLE entities ADD COLUMN subcategory TEXT DEFAULT 'Unrivaled';
            END $$;
        """,
        "sqlite_sql": """
            -- Check if column exists in SQLite
            PRAGMA table_info(entities);
            -- Add column if it doesn't exist (we'll handle this in Python code)
        """
    },
    {
        "version": "1.0.2",
        "description": "Add domain field to entities",
        "sql": """
            -- For PostgreSQL
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name='entities' AND column_name='domain'
                ) THEN
                    ALTER TABLE entities ADD COLUMN domain TEXT DEFAULT 'Sports';
                    
                    -- Update existing entities based on category
                    UPDATE entities SET domain = 'Sports' WHERE category = 'Sports';
                    UPDATE entities SET domain = 'Crypto' WHERE category = 'Crypto';
                END IF;
            EXCEPTION WHEN others THEN
                -- This will run if we're on SQLite
                ALTER TABLE entities ADD COLUMN domain TEXT DEFAULT 'Sports';
                -- Update existing entities
                UPDATE entities SET domain = 'Sports' WHERE category = 'Sports';
                UPDATE entities SET domain = 'Crypto' WHERE category = 'Crypto';
            END $$;
        """,
        "sqlite_sql": """
            -- Check if column exists in SQLite
            PRAGMA table_info(entities);
            -- Add column if it doesn't exist (we'll handle this in Python code)
            -- Then update values
            UPDATE entities SET domain = 'Sports' WHERE category = 'Sports';
            UPDATE entities SET domain = 'Crypto' WHERE category = 'Crypto';
        """
    },
    {
        "version": "1.0.3",
        "description": "Add updated_at and metadata fields",
        "sql": """
            -- Check if updated_at column exists
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name='entities' AND column_name='updated_at'
                ) THEN
                    ALTER TABLE entities ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                END IF;
            END $$;
            
            -- Add trigger function if it doesn't exist
            CREATE OR REPLACE FUNCTION update_modified_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            
            -- Create trigger
            DROP TRIGGER IF EXISTS update_entities_updated_at ON entities;
            CREATE TRIGGER update_entities_updated_at
            BEFORE UPDATE ON entities
            FOR EACH ROW
            EXECUTE FUNCTION update_modified_column();
            
            -- Check if metadata column exists
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name='entities' AND column_name='metadata'
                ) THEN
                    ALTER TABLE entities ADD COLUMN metadata JSONB DEFAULT '{}'::jsonb;
                END IF;
            END $$;
        """,
        "sqlite_sql": """
            -- Check if columns exist in SQLite
            PRAGMA table_info(entities);
            -- Add columns if they don't exist (we'll handle this in Python code)
            -- Create trigger for updated_at (we'll handle this separately)
        """
    },
    {
        "version": "1.0.4",
        "description": "Create indexes for common queries",
        "sql": """
            -- Create indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category);
            CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
            CREATE INDEX IF NOT EXISTS idx_hype_scores_timestamp ON hype_scores(timestamp);
            CREATE INDEX IF NOT EXISTS idx_hype_scores_entity_id ON hype_scores(entity_id);
            CREATE INDEX IF NOT EXISTS idx_component_metrics_entity_id ON component_metrics(entity_id);
            CREATE INDEX IF NOT EXISTS idx_component_metrics_metric_type ON component_metrics(metric_type);
        """,
        "sqlite_sql": """
            -- Create indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category);
            CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
            CREATE INDEX IF NOT EXISTS idx_hype_scores_timestamp ON hype_scores(timestamp);
            CREATE INDEX IF NOT EXISTS idx_hype_scores_entity_id ON hype_scores(entity_id);
            CREATE INDEX IF NOT EXISTS idx_component_metrics_entity_id ON component_metrics(entity_id);
            CREATE INDEX IF NOT EXISTS idx_component_metrics_metric_type ON component_metrics(metric_type);
        """
    },
    {
        "version": "1.0.5",
        "description": "Create entity relationships table",
        "sql": """
            -- Create the relationships table
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id SERIAL PRIMARY KEY,
                source_entity_id INTEGER NOT NULL REFERENCES entities(id),
                target_entity_id INTEGER NOT NULL REFERENCES entities(id),
                relationship_type TEXT NOT NULL,
                strength FLOAT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- Prevent duplicate relationships
                UNIQUE(source_entity_id, target_entity_id, relationship_type)
            );
        
            -- Create indexes for efficient lookups
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON entity_relationships(source_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON entity_relationships(target_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON entity_relationships(relationship_type);
        
            -- Create trigger for updated_at
            CREATE OR REPLACE FUNCTION update_entity_relationship_timestamp()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        
            DROP TRIGGER IF EXISTS update_entity_relationship_timestamp ON entity_relationships;
            CREATE TRIGGER update_entity_relationship_timestamp
            BEFORE UPDATE ON entity_relationships
            FOR EACH ROW
            EXECUTE FUNCTION update_entity_relationship_timestamp();
        """,
        "sqlite_sql": """
            -- Create the relationships table for SQLite
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_entity_id INTEGER NOT NULL,
                target_entity_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id),
                UNIQUE(source_entity_id, target_entity_id, relationship_type)
            );
        
            -- Create indexes for SQLite
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON entity_relationships(source_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON entity_relationships(target_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON entity_relationships(relationship_type);
        
            -- Create trigger for updated_at in SQLite
            CREATE TRIGGER IF NOT EXISTS update_entity_relationship_timestamp
            AFTER UPDATE ON entity_relationships
            FOR EACH ROW
            BEGIN
                UPDATE entity_relationships SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """
    }
]

def run_migrations():
    """Run all pending migrations."""
    # Ensure migrations table exists
    if not check_migrations_table():
        logger.error("Failed to create migrations table. Aborting.")
        migration_logger.error("Failed to create migrations table. Aborting.")
        return False
    
    # Get already applied migrations
    applied = get_applied_migrations()
    logger.info(f"Found {len(applied)} previously applied migrations")
    migration_logger.info(f"Found {len(applied)} previously applied migrations")
    
    # Determine if we're using SQLite
    from db_pool import SQLITE_AVAILABLE, POSTGRESQL_AVAILABLE
    using_sqlite = (POSTGRESQL_AVAILABLE is False or POSTGRESQL_AVAILABLE == "SQLITE")
    logger.info(f"Database type: {'SQLite' if using_sqlite else 'PostgreSQL'}")
    migration_logger.info(f"Database type: {'SQLite' if using_sqlite else 'PostgreSQL'}")
    
    # Apply pending migrations
    success = True
    for migration in MIGRATIONS:
        version = migration["version"]
        
        if version in applied:
            logger.info(f"Migration {version} already applied, skipping")
            migration_logger.info(f"Migration {version} already applied, skipping")
            continue
            
        logger.info(f"Applying migration {version}: {migration['description']}")
        migration_logger.info(f"Applying migration {version}: {migration['description']}")
        
        try:
            with DatabaseConnection() as conn:
                cursor = conn.cursor()
                
                # Choose the right SQL based on database type
                sql = migration["sqlite_sql"] if using_sqlite else migration["sql"]
                
                if using_sqlite and "sqlite_sql" in migration:
                    # For SQLite, we need special handling for schema changes
                    if version == "1.0.0":
                        # Check if category column exists
                        cursor.execute("PRAGMA table_info(entities)")
                        columns = [column[1] for column in cursor.fetchall()]
                        if "category" not in columns:
                            cursor.execute("ALTER TABLE entities ADD COLUMN category TEXT DEFAULT 'Sports'")
                    
                    elif version == "1.0.1":
                        # Check if subcategory column exists
                        cursor.execute("PRAGMA table_info(entities)")
                        columns = [column[1] for column in cursor.fetchall()]
                        if "subcategory" not in columns:
                            cursor.execute("ALTER TABLE entities ADD COLUMN subcategory TEXT DEFAULT 'Unrivaled'")
                    
                    elif version == "1.0.2":
                        # Check if domain column exists
                        cursor.execute("PRAGMA table_info(entities)")
                        columns = [column[1] for column in cursor.fetchall()]
                        if "domain" not in columns:
                            cursor.execute("ALTER TABLE entities ADD COLUMN domain TEXT DEFAULT 'Sports'")
                            # Update values
                            cursor.execute("UPDATE entities SET domain = 'Sports' WHERE category = 'Sports'")
                            cursor.execute("UPDATE entities SET domain = 'Crypto' WHERE category = 'Crypto'")
                    
                    elif version == "1.0.3":
                        # Check if updated_at and metadata columns exist
                        cursor.execute("PRAGMA table_info(entities)")
                        columns = [column[1] for column in cursor.fetchall()]
                        if "updated_at" not in columns:
                            cursor.execute("ALTER TABLE entities ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                        if "metadata" not in columns:
                            cursor.execute("ALTER TABLE entities ADD COLUMN metadata TEXT DEFAULT '{}'")
                            
                        # Create trigger for updated_at
                        cursor.execute("""
                            CREATE TRIGGER IF NOT EXISTS update_entities_updated_at
                            AFTER UPDATE ON entities
                            FOR EACH ROW
                            BEGIN
                                UPDATE entities SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                            END;
                        """)
                    
                    elif version == "1.0.4":
                        # Create indexes
                        cursor.execute(sql)
                
                else:
                    # For PostgreSQL or simple SQLite operations
                    cursor.execute(sql)
                
                # Commit the changes
                conn.commit()
                
                # Record the migration
                if not record_migration(version, migration["description"]):
                    logger.warning(f"Failed to record migration {version}, but changes were applied")
                
                logger.info(f"Successfully applied migration {version}")
                migration_logger.info(f"Successfully applied migration {version}")
        except Exception as e:
            logger.error(f"Error applying migration {version}: {e}")
            migration_logger.error(f"Error applying migration {version}: {e}") 
            success = False
            break
    
    if success:
        logger.info("All migrations completed successfully")
        migration_logger.info("All migrations completed successfully")
    else:
        logger.error("Migration process encountered errors")
        migration_logger.error("Migration process encountered errors") 
        
    return success

# Run migrations when directly executed
if __name__ == "__main__":
    logger.info("Starting database migrations...")
    migration_logger.info("Starting database migrations...")
    run_migrations()
# schema_manager.py - Customer schema management for HypeTorch
import os
import logging
from typing import Optional, Dict, List, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('schema_manager')

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
CONNECTION_TIMEOUT = 30

def get_direct_connection():
    """Get a direct database connection without schema setup."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    return psycopg2.connect(DATABASE_URL, connect_timeout=CONNECTION_TIMEOUT)

def execute_schema_query(query: str, params=None, fetch=True):
    """Execute a query against the database with manual connection management."""
    conn = None
    try:
        conn = get_direct_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if params is None:
                cursor.execute(query)
            else:
                cursor.execute(query, params)
            
            if fetch:
                result = cursor.fetchall()
                return [dict(row) for row in result]
            else:
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Schema query error: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def list_customer_schemas() -> List[str]:
    """List all customer schemas in the database."""
    try:
        query = """
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name LIKE 'customer_%' 
            ORDER BY schema_name
        """
        result = execute_schema_query(query)
        return [row['schema_name'] for row in result]
    except Exception as e:
        logger.error(f"Error listing customer schemas: {e}")
        return []

def schema_exists(schema_name: str) -> bool:
    """Check if a schema exists."""
    try:
        query = """
            SELECT COUNT(*) as count
            FROM information_schema.schemata 
            WHERE schema_name = %s
        """
        result = execute_schema_query(query, (schema_name,))
        return result[0]['count'] > 0
    except Exception as e:
        logger.error(f"Error checking schema existence: {e}")
        return False

def get_master_table_structure() -> Dict[str, str]:
    """Get the table structure from the public/pilot schema to replicate."""
    try:
        # Use the current DB_ENVIRONMENT or default to 'public'
        master_schema = os.environ.get("DB_ENVIRONMENT", "public")
        
        query = """
            SELECT table_name, 
                   array_to_string(
                       array_agg(
                           column_name || ' ' || 
                           data_type ||
                           CASE 
                               WHEN character_maximum_length IS NOT NULL 
                               THEN '(' || character_maximum_length || ')'
                               WHEN numeric_precision IS NOT NULL 
                               THEN '(' || numeric_precision || 
                                    CASE WHEN numeric_scale IS NOT NULL 
                                         THEN ',' || numeric_scale 
                                         ELSE '' 
                                    END || ')'
                               ELSE ''
                           END ||
                           CASE WHEN is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END
                           ORDER BY ordinal_position
                       ), 
                       ', '
                   ) as columns
            FROM information_schema.columns
            WHERE table_schema = %s
            AND table_name IN ('entities', 'current_metrics', 'historical_metrics', 
                              'hype_scores', 'api_keys', 'system_settings',
                              'token_transactions')
            GROUP BY table_name
            ORDER BY table_name
        """
        
        result = execute_schema_query(query, (master_schema,))
        
        # Convert to dictionary
        table_structures = {}
        for row in result:
            table_structures[row['table_name']] = row['columns']
        
        return table_structures
        
    except Exception as e:
        logger.error(f"Error getting master table structure: {e}")
        return {}

def create_customer_schema(customer_id: str, overwrite: bool = False) -> Tuple[bool, str]:
    """
    Create a new customer schema with all necessary tables.
    
    Args:
        customer_id: Customer identifier (e.g., 'customer_abc')
        overwrite: Whether to drop existing schema if it exists
    
    Returns:
        (success, message) tuple
    """
    try:
        schema_name = f"customer_{customer_id}" if not customer_id.startswith('customer_') else customer_id
        
        # Check if schema already exists
        if schema_exists(schema_name):
            if not overwrite:
                return False, f"Schema '{schema_name}' already exists. Use overwrite=True to replace."
            else:
                # Drop existing schema
                logger.info(f"Dropping existing schema: {schema_name}")
                drop_query = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
                execute_schema_query(drop_query, fetch=False)
        
        # Create the schema
        logger.info(f"Creating schema: {schema_name}")
        create_schema_query = f"CREATE SCHEMA {schema_name}"
        execute_schema_query(create_schema_query, fetch=False)
        
        # Get table structures from master schema
        table_structures = get_master_table_structure()
        if not table_structures:
            return False, "Failed to get master table structures"
        
        # Create tables in the new schema
        conn = get_direct_connection()
        try:
            with conn.cursor() as cursor:
                # Set search path to the new schema
                cursor.execute(f"SET search_path TO {schema_name}")
                
                # Create entities table
                cursor.execute(f"""
                    CREATE TABLE entities (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(200) NOT NULL UNIQUE,
                        type VARCHAR(50) DEFAULT 'person',
                        category VARCHAR(100) DEFAULT 'Sports',
                        subcategory VARCHAR(100) DEFAULT 'Unrivaled',
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create current_metrics table
                cursor.execute(f"""
                    CREATE TABLE current_metrics (
                        id SERIAL PRIMARY KEY,
                        entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                        metric_type VARCHAR(50) NOT NULL,
                        value DECIMAL(10,4),
                        time_period VARCHAR(50) DEFAULT 'last_30_days',
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(entity_id, metric_type)
                    )
                """)
                
                # Create historical_metrics table
                cursor.execute(f"""
                    CREATE TABLE historical_metrics (
                        id SERIAL PRIMARY KEY,
                        entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                        metric_type VARCHAR(50) NOT NULL,
                        value DECIMAL(10,4),
                        time_period VARCHAR(50) DEFAULT 'last_30_days',
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create hype_scores table (for backward compatibility)
                cursor.execute(f"""
                    CREATE TABLE hype_scores (
                        id SERIAL PRIMARY KEY,
                        entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                        score DECIMAL(10,4) NOT NULL,
                        time_period VARCHAR(50) DEFAULT 'last_30_days',
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create pipeline_runs table for tracking
                cursor.execute(f"""
                    CREATE TABLE pipeline_runs (
                        id SERIAL PRIMARY KEY,
                        customer_id VARCHAR(50) NOT NULL,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        status VARCHAR(20) DEFAULT 'running',
                        episodes_processed INTEGER DEFAULT 0,
                        entities_updated INTEGER DEFAULT 0,
                        error_message TEXT
                    )
                """)
                
                # Create indexes
                cursor.execute(f"CREATE INDEX idx_{schema_name}_entities_category ON entities(category)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_entities_subcategory ON entities(subcategory)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_current_metrics_entity ON current_metrics(entity_id)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_current_metrics_type ON current_metrics(metric_type)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_historical_metrics_entity ON historical_metrics(entity_id)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_historical_metrics_timestamp ON historical_metrics(timestamp)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_hype_scores_entity ON hype_scores(entity_id)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_hype_scores_timestamp ON hype_scores(timestamp)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_pipeline_runs_customer ON pipeline_runs(customer_id)")
                cursor.execute(f"CREATE INDEX idx_{schema_name}_pipeline_runs_status ON pipeline_runs(status)")
                
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
        logger.info(f"Successfully created customer schema: {schema_name}")
        return True, f"Schema '{schema_name}' created successfully with all tables and indexes"
        
    except Exception as e:
        logger.error(f"Error creating customer schema: {e}")
        return False, str(e)

def delete_customer_schema(customer_id: str) -> Tuple[bool, str]:
    """
    Delete a customer schema and all its data.
    
    Args:
        customer_id: Customer identifier
    
    Returns:
        (success, message) tuple
    """
    try:
        schema_name = f"customer_{customer_id}" if not customer_id.startswith('customer_') else customer_id
        
        if not schema_exists(schema_name):
            return False, f"Schema '{schema_name}' does not exist"
        
        # Drop the schema
        logger.info(f"Dropping schema: {schema_name}")
        drop_query = f"DROP SCHEMA {schema_name} CASCADE"
        execute_schema_query(drop_query, fetch=False)
        
        return True, f"Schema '{schema_name}' deleted successfully"
        
    except Exception as e:
        logger.error(f"Error deleting customer schema: {e}")
        return False, str(e)

def get_schema_statistics(schema_name: str) -> Dict:
    """Get statistics about a customer schema."""
    try:
        if not schema_exists(schema_name):
            return {"error": "Schema does not exist"}
        
        stats = {}
        
        # Get table counts
        tables = ['entities', 'current_metrics', 'historical_metrics', 'hype_scores', 'pipeline_runs']
        
        for table in tables:
            try:
                query = f"SELECT COUNT(*) as count FROM {schema_name}.{table}"
                result = execute_schema_query(query)
                stats[f"{table}_count"] = result[0]['count']
            except:
                stats[f"{table}_count"] = 0
        
        # Get latest pipeline run
        try:
            query = f"""
                SELECT status, start_time, end_time, episodes_processed, entities_updated
                FROM {schema_name}.pipeline_runs 
                ORDER BY start_time DESC 
                LIMIT 1
            """
            result = execute_schema_query(query)
            if result:
                stats["latest_pipeline_run"] = dict(result[0])
            else:
                stats["latest_pipeline_run"] = None
        except:
            stats["latest_pipeline_run"] = None
        
        # Get schema size
        try:
            query = """
                SELECT 
                    schemaname,
                    pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))::bigint) as size
                FROM pg_tables 
                WHERE schemaname = %s
                GROUP BY schemaname
            """
            result = execute_schema_query(query, (schema_name,))
            if result:
                stats["schema_size"] = result[0]['size']
            else:
                stats["schema_size"] = "0 bytes"
        except:
            stats["schema_size"] = "unknown"
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting schema statistics: {e}")
        return {"error": str(e)}

def clone_entities_to_schema(source_schema: str, target_schema: str, entity_names: List[str] = None) -> Tuple[bool, str]:
    """
    Clone entities from one schema to another.
    
    Args:
        source_schema: Source schema name
        target_schema: Target schema name  
        entity_names: Optional list of specific entities to clone
    
    Returns:
        (success, message) tuple
    """
    try:
        if not schema_exists(source_schema):
            return False, f"Source schema '{source_schema}' does not exist"
        
        if not schema_exists(target_schema):
            return False, f"Target schema '{target_schema}' does not exist"
        
        conn = get_direct_connection()
        entities_cloned = 0
        
        try:
            with conn.cursor() as cursor:
                # Build WHERE clause for specific entities
                where_clause = ""
                params = []
                
                if entity_names:
                    placeholders = ', '.join(['%s'] * len(entity_names))
                    where_clause = f" WHERE name IN ({placeholders})"
                    params = entity_names
                
                # Get entities from source schema
                query = f"""
                    SELECT name, type, category, subcategory, metadata
                    FROM {source_schema}.entities{where_clause}
                    ORDER BY name
                """
                
                cursor.execute(query, params)
                source_entities = cursor.fetchall()
                
                # Insert entities into target schema
                for entity_row in source_entities:
                    name, entity_type, category, subcategory, metadata = entity_row
                    
                    # Check if entity already exists in target
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {target_schema}.entities WHERE name = %s
                    """, (name,))
                    
                    if cursor.fetchone()[0] > 0:
                        continue  # Skip if exists
                    
                    # Insert entity
                    cursor.execute(f"""
                        INSERT INTO {target_schema}.entities 
                        (name, type, category, subcategory, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (name, entity_type, category, subcategory, metadata))
                    
                    entities_cloned += 1
                
                conn.commit()
                
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        
        return True, f"Cloned {entities_cloned} entities from '{source_schema}' to '{target_schema}'"
        
    except Exception as e:
        logger.error(f"Error cloning entities: {e}")
        return False, str(e)

def backup_customer_schema(schema_name: str, backup_file: str) -> Tuple[bool, str]:
    """
    Create a backup of a customer schema.
    
    Args:
        schema_name: Schema to backup
        backup_file: Path to backup file
    
    Returns:
        (success, message) tuple
    """
    try:
        if not schema_exists(schema_name):
            return False, f"Schema '{schema_name}' does not exist"
        
        import subprocess
        
        # Get database connection parameters
        db_url = DATABASE_URL
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "")
        
        # Parse connection string
        if "@" in db_url:
            auth, host_db = db_url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth
                password = ""
            
            if "/" in host_db:
                host_port, database = host_db.split("/", 1)
                # Remove any query parameters
                database = database.split("?")[0]
            else:
                host_port = host_db
                database = ""
                
            if ":" in host_port:
                host, port = host_port.split(":", 1)
            else:
                host = host_port
                port = "5432"
        else:
            return False, "Invalid database URL format"
        
        # Run pg_dump for the specific schema
        cmd = [
            "pg_dump",
            f"--host={host}",
            f"--port={port}",
            f"--username={username}",
            f"--dbname={database}",
            f"--schema={schema_name}",
            "--no-password",
            "--verbose",
            f"--file={backup_file}"
        ]
        
        env = os.environ.copy()
        if password:
            env["PGPASSWORD"] = password
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, f"Schema '{schema_name}' backed up to '{backup_file}'"
        else:
            return False, f"Backup failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"Error backing up schema: {e}")
        return False, str(e)

def restore_customer_schema(schema_name: str, backup_file: str) -> Tuple[bool, str]:
    """
    Restore a customer schema from backup.
    
    Args:
        schema_name: Schema to restore to
        backup_file: Path to backup file
    
    Returns:
        (success, message) tuple
    """
    try:
        if not os.path.exists(backup_file):
            return False, f"Backup file '{backup_file}' does not exist"
        
        import subprocess
        
        # Parse database URL (same as backup function)
        db_url = DATABASE_URL
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "")
        
        if "@" in db_url:
            auth, host_db = db_url.split("@", 1)
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth
                password = ""
            
            if "/" in host_db:
                host_port, database = host_db.split("/", 1)
                database = database.split("?")[0]
            else:
                host_port = host_db
                database = ""
                
            if ":" in host_port:
                host, port = host_port.split(":", 1)
            else:
                host = host_port
                port = "5432"
        else:
            return False, "Invalid database URL format"
        
        # Drop existing schema if it exists
        if schema_exists(schema_name):
            drop_success, drop_msg = delete_customer_schema(schema_name)
            if not drop_success:
                return False, f"Failed to drop existing schema: {drop_msg}"
        
        # Run psql to restore the schema
        cmd = [
            "psql",
            f"--host={host}",
            f"--port={port}",
            f"--username={username}",
            f"--dbname={database}",
            "--no-password",
            f"--file={backup_file}"
        ]
        
        env = os.environ.copy()
        if password:
            env["PGPASSWORD"] = password
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True, f"Schema '{schema_name}' restored from '{backup_file}'"
        else:
            return False, f"Restore failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"Error restoring schema: {e}")
        return False, str(e)

# CLI interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HypeTorch Schema Manager")
    parser.add_argument("action", choices=["create", "delete", "list", "stats", "clone", "backup", "restore"])
    parser.add_argument("--customer-id", help="Customer ID")
    parser.add_argument("--schema", help="Schema name")
    parser.add_argument("--source-schema", help="Source schema for cloning")
    parser.add_argument("--backup-file", help="Backup file path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing schema")
    
    args = parser.parse_args()
    
    if args.action == "create":
        if not args.customer_id:
            print("--customer-id required for create action")
            exit(1)
        success, message = create_customer_schema(args.customer_id, args.overwrite)
        print(f"{'Success' if success else 'Error'}: {message}")
    
    elif args.action == "delete":
        if not args.customer_id:
            print("--customer-id required for delete action")
            exit(1)
        success, message = delete_customer_schema(args.customer_id)
        print(f"{'Success' if success else 'Error'}: {message}")
    
    elif args.action == "list":
        schemas = list_customer_schemas()
        print("Customer schemas:")
        for schema in schemas:
            print(f"  {schema}")
    
    elif args.action == "stats":
        if not args.schema:
            print("--schema required for stats action")
            exit(1)
        stats = get_schema_statistics(args.schema)
        print(f"Statistics for {args.schema}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.action == "backup":
        if not args.schema or not args.backup_file:
            print("--schema and --backup-file required for backup action")
            exit(1)
        success, message = backup_customer_schema(args.schema, args.backup_file)
        print(f"{'Success' if success else 'Error'}: {message}")
    
    elif args.action == "restore":
        if not args.schema or not args.backup_file:
            print("--schema and --backup-file required for restore action")
            exit(1)
        success, message = restore_customer_schema(args.schema, args.backup_file)
        print(f"{'Success' if success else 'Error'}: {message}")
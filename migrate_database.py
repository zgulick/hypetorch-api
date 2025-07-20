# migrate_database.py
import os
import json
import sys
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("migration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('migration')

# Load environment variables
load_dotenv()

# Make sure PostgreSQL driver is available
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
    logger.info("✅ PostgreSQL driver found")
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.error("❌ PostgreSQL driver not found. Please install psycopg2-binary.")
    sys.exit(1)

# Get database URL from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.error("❌ DATABASE_URL not found in environment variables")
    sys.exit(1)

logger.info(f"✅ Found DATABASE_URL: {DATABASE_URL[:10]}...{DATABASE_URL[-10:]}")

# Database environment
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")
logger.info(f"✅ Using database environment: {DB_ENVIRONMENT}")

def connect_to_db():
    """Create a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set the search path to the appropriate schema
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {DB_ENVIRONMENT}")
        cursor.execute(f"SET search_path TO {DB_ENVIRONMENT}")
        conn.commit()
        
        return conn, cursor
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        sys.exit(1)

def get_current_entities():
    """Get all entities from the current database."""
    try:
        conn, cursor = connect_to_db()
        cursor.execute(f"SELECT * FROM {DB_ENVIRONMENT}.entities")
        entities = cursor.fetchall()
        cursor.close()
        conn.close()
        return entities
    except Exception as e:
        logger.error(f"❌ Error fetching current entities: {e}")
        return []

def get_current_metrics():
    """Get all metrics from the current database."""
    try:
        conn, cursor = connect_to_db()
        # Extract from component_metrics
        cursor.execute(f"""
            SELECT entity_id, metric_type, value, timestamp, time_period
            FROM {DB_ENVIRONMENT}.component_metrics
        """)
        component_metrics = cursor.fetchall()
        
        # Extract from hype_scores
        cursor.execute(f"""
            SELECT entity_id, score, timestamp, time_period, algorithm_version
            FROM {DB_ENVIRONMENT}.hype_scores
        """)
        hype_scores = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return {"component_metrics": component_metrics, "hype_scores": hype_scores}
    except Exception as e:
        logger.error(f"❌ Error fetching current metrics: {e}")
        return {"component_metrics": [], "hype_scores": []}

def migrate_data():
    """Main migration function that orchestrates the entire process."""
    start_time = time.time()
    logger.info("🚀 Starting database migration")
    
    # Step 1: Get current data
    logger.info("Step 1: Gathering current data")
    entities = get_current_entities()
    logger.info(f"✅ Found {len(entities)} entities in current database")
    
    metrics = get_current_metrics()
    logger.info(f"✅ Found {len(metrics['component_metrics'])} component metrics")
    logger.info(f"✅ Found {len(metrics['hype_scores'])} hype scores")
    
    # Step 2: Create new schema tables
    logger.info("Step 2: Setting up new schema tables")
    
    try:
        conn, cursor = connect_to_db()
        
        # Create the new tables according to the schema
        # Make sure these match your schema.sql design
        cursor.execute(f"""
            -- Create entities table if not exists
            CREATE TABLE IF NOT EXISTS {DB_ENVIRONMENT}.entities_new (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL DEFAULT 'person',
                category TEXT,
                subcategory TEXT,
                domain TEXT,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create hype_scores table if not exists
            CREATE TABLE IF NOT EXISTS {DB_ENVIRONMENT}.hype_scores_new (
                id SERIAL PRIMARY KEY,
                entity_id INTEGER NOT NULL,
                score FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_period TEXT,
                algorithm_version TEXT,
                FOREIGN KEY (entity_id) REFERENCES {DB_ENVIRONMENT}.entities_new(id)
            );
            
            -- Create metrics table if not exists
            CREATE TABLE IF NOT EXISTS {DB_ENVIRONMENT}.metrics_new (
                id SERIAL PRIMARY KEY,
                entity_id INTEGER NOT NULL,
                metric_type TEXT NOT NULL,
                value FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_period TEXT,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                FOREIGN KEY (entity_id) REFERENCES {DB_ENVIRONMENT}.entities_new(id)
            );
            
            -- Create entity_relationships table if not exists
            CREATE TABLE IF NOT EXISTS {DB_ENVIRONMENT}.entity_relationships_new (
                id SERIAL PRIMARY KEY,
                source_entity_id INTEGER NOT NULL,
                target_entity_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                strength FLOAT,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES {DB_ENVIRONMENT}.entities_new(id),
                FOREIGN KEY (target_entity_id) REFERENCES {DB_ENVIRONMENT}.entities_new(id),
                UNIQUE(source_entity_id, target_entity_id, relationship_type)
            );
            
            -- Create historical_data table for time series
            CREATE TABLE IF NOT EXISTS {DB_ENVIRONMENT}.historical_data_new (
                id SERIAL PRIMARY KEY,
                entity_id INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                value FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                time_period TEXT,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                FOREIGN KEY (entity_id) REFERENCES {DB_ENVIRONMENT}.entities_new(id)
            );
        """)
        
        conn.commit()
        logger.info("✅ Created new tables successfully")
        
        # Step 3: Migrate entities
        logger.info("Step 3: Migrating entities to new table")
        
        for entity in entities:
            # Map entity fields to new schema
            cursor.execute(f"""
                INSERT INTO {DB_ENVIRONMENT}.entities_new
                (id, name, type, category, subcategory, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (
                entity["id"],
                entity["name"],
                entity["type"],
                entity["category"],
                entity["subcategory"],
                entity["created_at"]
            ))
        
        conn.commit()
        logger.info(f"✅ Migrated {len(entities)} entities")
        
        # Step 4: Migrate metrics
        logger.info("Step 4: Migrating metrics to new tables")
        
        # Migrate component metrics
        for metric in metrics["component_metrics"]:
            cursor.execute(f"""
                INSERT INTO {DB_ENVIRONMENT}.metrics_new
                (entity_id, metric_type, value, timestamp, time_period)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                metric["entity_id"],
                metric["metric_type"],
                metric["value"],
                metric["timestamp"],
                metric["time_period"]
            ))
        
        # Migrate hype scores
        for score in metrics["hype_scores"]:
            cursor.execute(f"""
                INSERT INTO {DB_ENVIRONMENT}.hype_scores_new
                (entity_id, score, timestamp, time_period, algorithm_version)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                score["entity_id"],
                score["score"],
                score["timestamp"],
                score["time_period"],
                score["algorithm_version"]
            ))
        
        conn.commit()
        logger.info(f"✅ Migrated {len(metrics['component_metrics'])} component metrics")
        logger.info(f"✅ Migrated {len(metrics['hype_scores'])} hype scores")
        
        # Step 5: Validate migration
        logger.info("Step 5: Validating migration")
        
        # Check entity counts
        cursor.execute(f"SELECT COUNT(*) as count FROM {DB_ENVIRONMENT}.entities_new")
        new_entity_count = cursor.fetchone()["count"]
        
        # Check metric counts
        cursor.execute(f"SELECT COUNT(*) as count FROM {DB_ENVIRONMENT}.metrics_new")
        new_metrics_count = cursor.fetchone()["count"]
        
        # Check hype score counts
        cursor.execute(f"SELECT COUNT(*) as count FROM {DB_ENVIRONMENT}.hype_scores_new")
        new_hype_scores_count = cursor.fetchone()["count"]
        
        logger.info(f"Validation Results:")
        logger.info(f"  Original entities: {len(entities)}, New entities: {new_entity_count}")
        logger.info(f"  Original component metrics: {len(metrics['component_metrics'])}, New metrics: {new_metrics_count}")
        logger.info(f"  Original hype scores: {len(metrics['hype_scores'])}, New hype scores: {new_hype_scores_count}")
        
        # Step 6: Finalize migration (rename tables)
        logger.info("Step 6: Finalizing migration")
        
        # Backup original tables first
        cursor.execute(f"""
            ALTER TABLE IF EXISTS {DB_ENVIRONMENT}.entities 
            RENAME TO entities_backup;
            
            ALTER TABLE IF EXISTS {DB_ENVIRONMENT}.component_metrics 
            RENAME TO component_metrics_backup;
            
            ALTER TABLE IF EXISTS {DB_ENVIRONMENT}.hype_scores 
            RENAME TO hype_scores_backup;

            ALTER TABLE IF EXISTS {DB_ENVIRONMENT}.entity_relationships 
            RENAME TO entity_relationships_backup;
        """)
        
        # Rename new tables to replace old ones
        cursor.execute(f"""
            ALTER TABLE {DB_ENVIRONMENT}.entities_new 
            RENAME TO entities;
            
            ALTER TABLE {DB_ENVIRONMENT}.metrics_new 
            RENAME TO metrics;
            
            ALTER TABLE {DB_ENVIRONMENT}.hype_scores_new 
            RENAME TO hype_scores;
            
            ALTER TABLE {DB_ENVIRONMENT}.entity_relationships_new 
            RENAME TO entity_relationships;
            
            ALTER TABLE {DB_ENVIRONMENT}.historical_data_new 
            RENAME TO historical_data;
        """)
        
        conn.commit()
        logger.info("✅ Successfully renamed tables to finalize migration")
        
        # Step 7: Create indexes for better performance
        logger.info("Step 7: Creating indexes")
        
        cursor.execute(f"""
            -- Create indexes on entities table
            CREATE INDEX IF NOT EXISTS idx_entities_name ON {DB_ENVIRONMENT}.entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON {DB_ENVIRONMENT}.entities(type);
            CREATE INDEX IF NOT EXISTS idx_entities_category ON {DB_ENVIRONMENT}.entities(category);
            
            -- Create indexes on metrics table
            CREATE INDEX IF NOT EXISTS idx_metrics_entity_id ON {DB_ENVIRONMENT}.metrics(entity_id);
            CREATE INDEX IF NOT EXISTS idx_metrics_type ON {DB_ENVIRONMENT}.metrics(metric_type);
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON {DB_ENVIRONMENT}.metrics(timestamp);
            
            -- Create indexes on hype_scores table
            CREATE INDEX IF NOT EXISTS idx_hype_scores_entity_id ON {DB_ENVIRONMENT}.hype_scores(entity_id);
            CREATE INDEX IF NOT EXISTS idx_hype_scores_timestamp ON {DB_ENVIRONMENT}.hype_scores(timestamp);
            
            -- Create indexes on historical_data table
            CREATE INDEX IF NOT EXISTS idx_historical_data_entity_id ON {DB_ENVIRONMENT}.historical_data(entity_id);
            CREATE INDEX IF NOT EXISTS idx_historical_data_type ON {DB_ENVIRONMENT}.historical_data(data_type);
            CREATE INDEX IF NOT EXISTS idx_historical_data_timestamp ON {DB_ENVIRONMENT}.historical_data(timestamp);
            
            -- Create indexes on entity_relationships table
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON {DB_ENVIRONMENT}.entity_relationships(source_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON {DB_ENVIRONMENT}.entity_relationships(target_entity_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON {DB_ENVIRONMENT}.entity_relationships(relationship_type);
        """)
        
        conn.commit()
        logger.info("✅ Successfully created indexes")
        
        cursor.close()
        conn.close()
        
        # Calculate total execution time
        execution_time = time.time() - start_time
        logger.info(f"✅ Migration completed successfully in {execution_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False

def import_latest_json():
    """Import the latest JSON data into the new schema."""
    try:
        # Find the latest JSON file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_file = os.path.join(base_dir, "hypetorch_latest_output.json")
        
        if not os.path.exists(json_file):
            logger.error(f"❌ JSON file not found: {json_file}")
            return False
            
        logger.info(f"📂 Found JSON file: {json_file}")
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Connect to database
        conn, cursor = connect_to_db()
        
        # Extract and store entities
        time_period = "last_30_days"  # Default
        timestamp = datetime.now()
        
        # Import HYPE scores
        if "hype_scores" in data:
            logger.info(f"Importing {len(data['hype_scores'])} HYPE scores")
            
            for entity_name, score in data["hype_scores"].items():
                # First, ensure the entity exists
                cursor.execute(f"""
                    INSERT INTO {DB_ENVIRONMENT}.entities 
                    (name, type, category, subcategory)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING id
                """, (
                    entity_name,
                    "person",  # Default type
                    "Sports",  # Default category
                    "Unrivaled"  # Default subcategory
                ))
                
                result = cursor.fetchone()
                if result:
                    entity_id = result["id"]
                else:
                    # Get the ID of the existing entity
                    cursor.execute(f"SELECT id FROM {DB_ENVIRONMENT}.entities WHERE name = %s", (entity_name,))
                    entity_id = cursor.fetchone()["id"]
                
                # Insert HYPE score
                cursor.execute(f"""
                    INSERT INTO {DB_ENVIRONMENT}.hype_scores
                    (entity_id, score, timestamp, time_period, algorithm_version)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    entity_id,
                    score,
                    timestamp,
                    time_period,
                    "1.0"  # Default version
                ))
        
        # Import other metrics
        metrics_mapping = {
            "talk_time_counts": "talk_time",
            "mention_counts": "mentions",
            "wikipedia_views": "wikipedia_views",
            "reddit_mentions": "reddit_mentions",
            "google_trends": "google_trends",
            "google_news_mentions": "google_news_mentions",
            "rodmn_scores": "rodmn_score"
        }
        
        for json_key, metric_type in metrics_mapping.items():
            if json_key in data:
                logger.info(f"Importing {len(data[json_key])} {metric_type} metrics")
                
                for entity_name, value in data[json_key].items():
                    # Get the entity ID
                    cursor.execute(f"SELECT id FROM {DB_ENVIRONMENT}.entities WHERE name = %s", (entity_name,))
                    result = cursor.fetchone()
                    
                    if result:
                        entity_id = result["id"]
                        
                        # Insert metric
                        cursor.execute(f"""
                            INSERT INTO {DB_ENVIRONMENT}.metrics
                            (entity_id, metric_type, value, timestamp, time_period)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            entity_id,
                            metric_type,
                            value,
                            timestamp,
                            time_period
                        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("✅ Successfully imported latest JSON data")
        return True
    except Exception as e:
        logger.error(f"❌ Error importing latest JSON: {e}")
        if 'conn' in locals():
            conn.rollback()
            cursor.close()
            conn.close()
        return False

if __name__ == "__main__":
    # Check if user wants to proceed
    print("⚠️ WARNING: This script will migrate your database to a new schema. ⚠️")
    print("It's recommended to back up your database before proceeding.")
    print("Would you like to continue? (y/n)")
    
    response = input().strip().lower()
    if response != 'y':
        print("Migration cancelled.")
        sys.exit(0)
    
    # Run the migration
    success = migrate_data()
    
    if success:
        print("\nWould you like to import the latest JSON data? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            import_success = import_latest_json()
            if import_success:
                print("✅ Migration and import completed successfully")
            else:
                print("✅ Migration completed successfully, but JSON import failed")
        else:
            print("✅ Migration completed successfully without JSON import")
    else:
        print("❌ Migration failed")
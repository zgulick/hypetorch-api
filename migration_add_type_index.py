#!/usr/bin/env python3
"""
Migration: Add index on entities.type column for Dynamic Vertical System

This migration adds a performance index on the entities.type column to support
efficient vertical filtering by entity type (person vs non-person).

Run with: python migration_add_type_index.py
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import execute_query, get_connection

load_dotenv()

def check_index_exists(schema_name):
    """Check if the type index already exists."""
    try:
        # Try to create the index - if it exists, it will fail gracefully
        query = f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_entities_type ON entities(type)"
        execute_query(query, fetch=False, schema_name=schema_name)
        return True
    except Exception as e:
        print(f"Note: {e}")
        return False

def add_type_index():
    """Add index on entities.type column."""

    # Get current schema/environment
    schema_name = os.environ.get("DB_ENVIRONMENT", "development")

    print(f"📊 Creating index on entities.type in schema: {schema_name}")

    try:
        # Create the index (IF NOT EXISTS handles duplicates)
        query = f"CREATE INDEX IF NOT EXISTS idx_{schema_name}_entities_type ON entities(type)"
        execute_query(query, fetch=False, schema_name=schema_name)

        print(f"✅ Successfully created/verified index idx_{schema_name}_entities_type")
        return True

    except Exception as e:
        print(f"❌ Error creating index: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Migration: Add entities.type index for Dynamic Vertical System")
    print("=" * 60)

    success = add_type_index()

    if success:
        print("\n✅ Migration completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

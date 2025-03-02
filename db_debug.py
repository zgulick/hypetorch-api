"""
Simple script to test PostgreSQL connection.
Run this after deployment with: python db_debug.py
"""

import os
import sys

print("Starting PostgreSQL connection test...")
print(f"Python version: {sys.version}")

# Try to import psycopg2
try:
    import psycopg2
    print(f"psycopg2 version: {psycopg2.__version__}")
except ImportError:
    print("❌ ERROR: psycopg2 is not installed!")
    sys.exit(1)

# Get DATABASE_URL from environment
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    print("❌ ERROR: DATABASE_URL is not set!")
    sys.exit(1)

print(f"DATABASE_URL is set: {database_url[:10]}...{database_url[-10:]}")

# Try to connect
try:
    conn = psycopg2.connect(database_url)
    print("✅ Successfully connected to PostgreSQL!")
    
    # Test basic query
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL version: {version[0]}")
    
    # Close connection
    conn.close()
except Exception as e:
    print(f"❌ ERROR connecting to PostgreSQL: {e}")
    sys.exit(1)

print("✅ PostgreSQL connection test completed successfully!")
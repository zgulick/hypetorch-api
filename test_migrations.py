# test_migrations.py
import os
import sqlite3
import tempfile
import pytest
from migrations import run_migrations, check_migrations_table, get_applied_migrations

# Create a temporary database for testing
@pytest.fixture
def test_db():
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.db')
    os.close(temp_fd)
    
    # Set environment variable to point to this test database
    original_db_url = os.environ.get('DATABASE_URL')
    os.environ['DATABASE_URL'] = f'sqlite:///{temp_path}'
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    if original_db_url:
        os.environ['DATABASE_URL'] = original_db_url
    else:
        del os.environ['DATABASE_URL']

def test_migrations_table(test_db):
    """Test that migrations table is created correctly"""
    # Run the function
    result = check_migrations_table()
    
    # Check that it returned success
    assert result == True
    
    # Connect to the DB and check that the table exists
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'")
    assert cursor.fetchone() is not None
    conn.close()

def test_run_migrations(test_db):
    """Test that migrations run successfully"""
    # Run migrations
    result = run_migrations()
    
    # Check result
    assert result == True
    
    # Check that migrations were recorded
    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM schema_migrations")
    count = cursor.fetchone()[0]
    assert count > 0, "No migrations were recorded"
    conn.close()

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
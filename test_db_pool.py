# test_db_pool.py
import time
import threading
import logging
from db_pool import execute_query, DatabaseConnection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger('test_db_pool')

def test_query():
    """Execute a simple query to test the connection."""
    try:
        # Simple SQL query that works in both PostgreSQL and SQLite
        result = execute_query("SELECT 1 AS test")
        logger.info(f"Query executed successfully: {result}")
        return True
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return False

def test_concurrent_connections():
    """Test concurrent connections to the database."""
    def worker(worker_id):
        logger.info(f"Worker {worker_id} starting")
        for i in range(3):
            try:
                with DatabaseConnection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 AS test")
                    result = cursor.fetchone()
                    logger.info(f"Worker {worker_id}, query {i+1}: {result}")
                    time.sleep(0.5)  # Simulate work
            except Exception as e:
                logger.error(f"Worker {worker_id} query failed: {e}")
        logger.info(f"Worker {worker_id} finished")

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def test_manual_transaction():
    """Test a manual transaction with multiple queries."""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 AS test1")
            result1 = cursor.fetchone()
            cursor.execute("SELECT 2 AS test2")
            result2 = cursor.fetchone()
            logger.info(f"Transaction results: {result1}, {result2}")
            return True
    except Exception as e:
        logger.error(f"Transaction failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Testing database connection pool...")
    test_query()
    test_concurrent_connections()
    test_manual_transaction()
    logger.info("Connection pool test completed")

if __name__ == "__main__":
    main()
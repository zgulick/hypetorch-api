# db_config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection settings
DATABASE_URL = os.environ.get("DATABASE_URL")
SQLITE_PATH = os.environ.get("SQLITE_PATH", "hypetorch.db")

# Determine server environment
if os.path.exists("/opt/render/project"):
    # Render environment
    BASE_DIR = "/opt/render/project/src"
    SQLITE_PATH = os.path.join(BASE_DIR, SQLITE_PATH)
else:
    # Local environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SQLITE_PATH = os.path.join(BASE_DIR, SQLITE_PATH)

# Connection pool settings
DB_MAX_CONNECTIONS = int(os.environ.get("DB_MAX_CONNECTIONS", "10"))
DB_MIN_CONNECTIONS = int(os.environ.get("DB_MIN_CONNECTIONS", "2"))
DB_MAX_IDLE_TIME = int(os.environ.get("DB_MAX_IDLE_TIME", "60"))  # seconds
DB_CONNECTION_TIMEOUT = int(os.environ.get("DB_CONNECTION_TIMEOUT", "30"))  # seconds

# Schema settings
DB_ENVIRONMENT = os.environ.get("DB_ENVIRONMENT", "development")

# Determine which database type to use
try:
    import psycopg2
    POSTGRESQL_AVAILABLE = True
    print("PostgreSQL driver available")
except ImportError:
    POSTGRESQL_AVAILABLE = False
    print("PostgreSQL driver not available, will use SQLite")

def get_db_settings():
    """Get database settings as a dictionary."""
    return {
        "postgresql_available": POSTGRESQL_AVAILABLE,
        "database_url": DATABASE_URL,
        "sqlite_path": SQLITE_PATH,
        "base_dir": BASE_DIR,
        "max_connections": DB_MAX_CONNECTIONS,
        "min_connections": DB_MIN_CONNECTIONS,
        "max_idle_time": DB_MAX_IDLE_TIME,
        "connection_timeout": DB_CONNECTION_TIMEOUT,
        "environment": DB_ENVIRONMENT
    }
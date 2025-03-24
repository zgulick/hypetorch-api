# run_migrations.py
from migrations import run_migrations

if __name__ == "__main__":
    print("Starting database migrations...")
    success = run_migrations()
    if success:
        print("✅ Migrations completed successfully!")
    else:
        print("❌ Migrations encountered errors - check logs for details")
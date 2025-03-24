# Database Migration System

## Overview

HypeTorch uses a versioned migration system to manage database schema changes. This ensures consistent database structure across all environments and allows for smooth upgrades.

## Migration Versions

Each migration has:
- **Version**: Semantic version number (e.g., "1.0.0")
- **Description**: Short text explaining the changes
- **SQL**: PostgreSQL SQL statements
- **SQLite SQL**: SQLite-compatible SQL statements (if needed)

## Current Migrations

1. **1.0.0**: Add category field to entities
2. **1.0.1**: Add subcategory field to entities
3. **1.0.2**: Add domain field to entities
4. **1.0.3**: Add updated_at and metadata fields
5. **1.0.4**: Create indexes for common queries

## Adding New Migrations

To add a new migration:

1. Add a new entry to the `MIGRATIONS` list in `migrations.py`
2. Follow the existing pattern for version, description, and SQL
3. Ensure compatibility with both PostgreSQL and SQLite
4. Test the migration on a development database
5. Run `python run_migrations.py` to apply the migration

## Deployment

Migrations run automatically on application startup in all environments:

1. The application checks for the existence of the `schema_migrations` table
2. It retrieves the list of previously applied migrations
3. It applies any pending migrations in version order
4. Results are logged to `migrations.log`

## Rollback

To rollback a migration:

1. Create a new migration that undoes the changes
2. Apply the new migration
3. Remove the previous migration from the `MIGRATIONS` list
4. Note: Data changes cannot be automatically rolled back
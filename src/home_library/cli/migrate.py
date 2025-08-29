"""Database migration CLI command.

This module provides CLI commands for managing the database schema,
including creating tables, resetting the database, and checking status.
"""

# ruff: noqa: T201
import argparse
import sys

from home_library.database import get_db_service


def create_tables() -> None:
    """Create all database tables."""
    try:
        db_service = get_db_service()
        db_service.create_tables()
        print("✅ Database tables created successfully!")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        sys.exit(1)


def drop_tables() -> None:
    """Drop all database tables."""
    try:
        db_service = get_db_service()
        db_service.drop_tables()
        print("✅ Database tables dropped successfully!")
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        sys.exit(1)


def reset_database() -> None:
    """Reset the database by dropping and recreating all tables."""
    try:
        db_service = get_db_service()
        db_service.reset_database()
        print("✅ Database reset successfully!")
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        sys.exit(1)


def check_status() -> None:
    """Check database status and connection."""
    try:
        db_service = get_db_service()

        if db_service.health_check():
            print("✅ Database connection: HEALTHY")
        else:
            print("❌ Database connection: UNHEALTHY")
            sys.exit(1)

        # Get database info
        info = db_service.get_database_info()
        print(f"📊 Database URL: {info['database_url']}")
        print(f"📊 Status: {info['status']}")

        if info["status"] == "healthy":
            print(f"📊 Database Size: {info.get('database_size', 'Unknown')}")
            if "tables" in info:
                print("📊 Tables:")
                for table in info["tables"]:
                    print(f"   - {table['tablename']}: {table['inserts']} inserts")

    except Exception as e:
        print(f"❌ Error checking database status: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Database migration and management commands"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create tables command
    subparsers.add_parser(
        "create", help="Create all database tables"
    )

    # Drop tables command
    subparsers.add_parser(
        "drop", help="Drop all database tables"
    )

    # Reset database command
    subparsers.add_parser(
        "reset", help="Reset database (drop and recreate all tables)"
    )

    # Status command
    subparsers.add_parser(
        "status", help="Check database status and connection"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    if args.command == "create":
        create_tables()
    elif args.command == "drop":
        drop_tables()
    elif args.command == "reset":
        reset_database()
    elif args.command == "status":
        check_status()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

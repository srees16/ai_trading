# Centurion Capital LLC - Database Setup Script
"""
Script to initialize the database schema and run initial setup.

Usage:
    python setup_database.py

Prerequisites:
    1. PostgreSQL with TimescaleDB extension installed
    2. Database created: CREATE DATABASE centurion_trading;
    3. TimescaleDB enabled: CREATE EXTENSION IF NOT EXISTS timescaledb;
    4. .env file configured with database credentials
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_configuration():
    """Check if database is properly configured."""
    if not Config.is_database_configured():
        logger.error("Database not configured!")
        logger.error(
            "CENTURION_DB_PASSWORD and CENTURION_DATABASE_URL are not set.\n"
            "  Run the export commands from README → Quick Start → Step 2 in this terminal first.\n"
            "  macOS/Linux:  export CENTURION_DB_PASSWORD='superadmin1' ...\n"
            "  Windows PS:   $env:CENTURION_DB_PASSWORD='superadmin1' ..."
        )
        return False
    
    logger.info(f"Database host: {Config.DB_HOST}")
    logger.info(f"Database name: {Config.DB_NAME}")
    logger.info(f"Database user: {Config.DB_USER}")
    return True


def initialize_database():
    """Initialize database tables and hypertables."""
    try:
        from database import DatabaseManager, get_database_service
        
        # Get database manager
        db_manager = DatabaseManager()
        
        # Check health
        if not db_manager.health_check():
            logger.error("Could not connect to database")
            logger.error("Make sure PostgreSQL is running and credentials are correct")
            return False
        
        logger.info(" Database connection successful")
        
        # Create tables
        if db_manager.initialize_database():
            logger.info(" Database tables created successfully")
        else:
            logger.error("Failed to create database tables")
            return False
        
        # Test the service layer
        service = get_database_service()
        if service.is_available:
            logger.info("Database service layer ready")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False


def show_table_info():
    """Display information about created tables."""
    try:
        from database import DatabaseManager
        from sqlalchemy import text
        
        db_manager = DatabaseManager()
        
        with db_manager.get_session() as session:
            # Get table information
            result = session.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            
            tables = [row[0] for row in result]
            
            if tables:
                logger.info("\n Database Tables:")
                for table in tables:
                    logger.info(f"   • {table}")
            
            # Check hypertables
            try:
                result = session.execute(text("""
                    SELECT hypertable_name 
                    FROM timescaledb_information.hypertables
                    WHERE hypertable_schema = 'public';
                """))
                hypertables = [row[0] for row in result]
                
                if hypertables:
                    logger.info("\n TimescaleDB Hypertables:")
                    for ht in hypertables:
                        logger.info(f"   • {ht}")
            except Exception:
                logger.info("\nTimescaleDB hypertables not found (extension may not be enabled)")
        
    except Exception as e:
        logger.error(f"Could not get table info: {e}")


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print(" Centurion Capital LLC - Database Setup")
    print("=" * 60 + "\n")
    
    # Check configuration
    if not check_configuration():
        print("\nSetup aborted. Please configure database settings.")
        return 1
    
    # Initialize database
    if not initialize_database():
        print("\nDatabase initialization failed.")
        return 1
    
    # Show table info
    show_table_info()
    
    print("\n" + "=" * 60)
    print(" Database setup completed successfully!")
    print("=" * 60)
    print("\nYou can now run: streamlit run app.py")
    print("Analysis results will be automatically saved to the database.\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Centurion Capital LLC - Database Connection Manager
"""
Enterprise-grade database connection management with connection pooling,
health checks, and automatic reconnection for PostgreSQL + TimescaleDB.
"""

import os
import logging
from typing import Optional, Generator
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""
    
    def __init__(self):
        self.host = os.getenv('CENTURION_DB_HOST', os.getenv('DB_HOST', 'localhost'))
        self.port = int(os.getenv('CENTURION_DB_PORT', os.getenv('DB_PORT', '5432')))
        self.database = os.getenv('CENTURION_DB_NAME', os.getenv('DB_NAME', 'centurion_trading'))
        self.username = os.getenv('CENTURION_DB_USER', os.getenv('DB_USER', 'centurion'))
        self.password = os.getenv('CENTURION_DB_PASSWORD', os.getenv('DB_PASSWORD', ''))
        self.pool_size = int(os.getenv('CENTURION_DB_POOL_SIZE', os.getenv('DB_POOL_SIZE', '10')))
        self.max_overflow = int(os.getenv('CENTURION_DB_MAX_OVERFLOW', os.getenv('DB_MAX_OVERFLOW', '20')))
        self.pool_timeout = int(os.getenv('CENTURION_DB_POOL_TIMEOUT', os.getenv('DB_POOL_TIMEOUT', '30')))
        self.pool_recycle = int(os.getenv('CENTURION_DB_POOL_RECYCLE', os.getenv('DB_POOL_RECYCLE', '3600')))
        self.echo_sql = os.getenv('DB_ECHO_SQL', 'false').lower() == 'true'
        self.ssl_mode = os.getenv('DB_SSL_MODE', 'prefer')
        
        # TimescaleDB specific
        self.enable_timescaledb = os.getenv('DB_ENABLE_TIMESCALEDB', 'true').lower() == 'true'
        self.chunk_interval = os.getenv('DB_CHUNK_INTERVAL', '7 days')
    
    @property
    def connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        password = quote_plus(self.password) if self.password else ''
        auth = f"{self.username}:{password}@" if self.username else ""
        return f"postgresql+psycopg2://{auth}{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Build async PostgreSQL connection string."""
        password = quote_plus(self.password) if self.password else ''
        auth = f"{self.username}:{password}@" if self.username else ""
        return f"postgresql+asyncpg://{auth}{self.host}:{self.port}/{self.database}"


class DatabaseManager:
    """
    Enterprise-grade database connection manager.
    
    Features:
    - Connection pooling with QueuePool
    - Automatic reconnection on failure
    - Health check support
    - TimescaleDB hypertable management
    - Thread-safe session management
    """
    
    _instance: Optional['DatabaseManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'DatabaseManager':
        """Singleton pattern for database manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database manager (only runs once due to singleton)."""
        if DatabaseManager._initialized:
            return
        
        self.config = DatabaseConfig()
        self._engine = None
        self._session_factory = None
        self._scoped_session = None
        DatabaseManager._initialized = True
        logger.info("DatabaseManager initialized")
    
    @property
    def engine(self):
        """Get or create SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Enable connection health check
                echo=self.config.echo_sql,
                connect_args={
                    'connect_timeout': 10,
                    'application_name': 'CenturionCapital',
                }
            )
            
            # Add event listeners for connection management
            @event.listens_for(self._engine, "connect")
            def on_connect(dbapi_conn, connection_record):
                logger.debug("New database connection established")
            
            @event.listens_for(self._engine, "checkout")
            def on_checkout(dbapi_conn, connection_record, connection_proxy):
                logger.debug("Connection checked out from pool")
            
            logger.info(f"Database engine created: {self.config.host}:{self.config.port}/{self.config.database}")
        
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
        return self._session_factory
    
    @property
    def scoped_session(self) -> scoped_session:
        """Get thread-local scoped session."""
        if self._scoped_session is None:
            self._scoped_session = scoped_session(self.session_factory)
        return self._scoped_session
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic cleanup.
        
        Usage:
            with db_manager.get_session() as session:
                session.query(Model).all()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error during database operation: {e}")
            raise
        finally:
            session.close()
    
    def health_check(self) -> dict:
        """
        Perform database health check.
        
        Returns:
            dict: Health check results with status and details
        """
        result = {
            'healthy': False,
            'database': self.config.database,
            'host': self.config.host,
            'port': self.config.port,
            'pool_size': self.config.pool_size,
            'timescaledb_enabled': False,
            'error': None
        }
        
        try:
            with self.get_session() as session:
                # Basic connectivity check
                session.execute(text("SELECT 1"))
                result['healthy'] = True
                
                # Check TimescaleDB extension
                if self.config.enable_timescaledb:
                    try:
                        ts_result = session.execute(
                            text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                        ).fetchone()
                        if ts_result:
                            result['timescaledb_enabled'] = True
                            result['timescaledb_version'] = ts_result[0]
                    except Exception:
                        pass
                
                # Get pool stats
                pool = self.engine.pool
                result['pool_status'] = {
                    'size': pool.size(),
                    'checked_in': pool.checkedin(),
                    'checked_out': pool.checkedout(),
                    'overflow': pool.overflow()
                }
                
        except OperationalError as e:
            result['error'] = f"Connection failed: {str(e)}"
            logger.error(f"Database health check failed: {e}")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Database health check error: {e}")
        
        return result
    
    def initialize_database(self) -> bool:
        """
        Initialize database schema and TimescaleDB extensions.
        
        Returns:
            bool: True if initialization successful
        """
        from database.models import Base
        
        try:
            with self.get_session() as session:
                # Enable TimescaleDB extension if configured
                if self.config.enable_timescaledb:
                    try:
                        session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
                        session.commit()
                        logger.info("TimescaleDB extension enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable TimescaleDB: {e}")
            
            # Create all tables
            Base.metadata.create_all(self.engine)
            logger.info("Database schema created successfully")
            
            # Create hypertables for time-series data
            self._create_hypertables()
            
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def _create_hypertables(self):
        """Create TimescaleDB hypertables for time-series tables."""
        if not self.config.enable_timescaledb:
            return
        
        hypertables = [
            ('stock_signals', 'created_at'),
            ('fundamental_metrics', 'recorded_at'),
            ('news_items', 'published_at'),
        ]
        
        with self.get_session() as session:
            for table_name, time_column in hypertables:
                try:
                    # Check if table exists and is not already a hypertable
                    exists = session.execute(
                        text(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = :table_name
                            )
                        """),
                        {'table_name': table_name}
                    ).scalar()
                    
                    if exists:
                        is_hypertable = session.execute(
                            text("""
                                SELECT EXISTS (
                                    SELECT FROM timescaledb_information.hypertables 
                                    WHERE hypertable_name = :table_name
                                )
                            """),
                            {'table_name': table_name}
                        ).scalar()
                        
                        if not is_hypertable:
                            session.execute(
                                text(f"""
                                    SELECT create_hypertable(
                                        :table_name, 
                                        :time_column,
                                        if_not_exists => TRUE,
                                        chunk_time_interval => INTERVAL :chunk_interval
                                    )
                                """),
                                {
                                    'table_name': table_name,
                                    'time_column': time_column,
                                    'chunk_interval': self.config.chunk_interval
                                }
                            )
                            logger.info(f"Created hypertable for {table_name}")
                    
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {table_name}: {e}")
    
    def close(self):
        """Close all connections and cleanup."""
        if self._scoped_session:
            self._scoped_session.remove()
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")
        
        DatabaseManager._instance = None
        DatabaseManager._initialized = False


# Global instance accessor
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """Get a database session from the global manager."""
    return get_db_manager().get_session()

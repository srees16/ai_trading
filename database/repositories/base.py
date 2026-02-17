# Centurion Capital LLC - Base Repository
"""
Base repository class with common CRUD operations.
"""

import logging
from typing import TypeVar, Generic, List, Optional, Type, Any, Dict
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_, or_, desc, asc

from database.models import Base

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T]):
    """
    Generic base repository with common database operations.
    
    Implements the Repository pattern for clean data access abstraction.
    """
    
    def __init__(self, session: Session, model_class: Type[T]):
        """
        Initialize repository.
        
        Args:
            session: SQLAlchemy session
            model_class: The model class this repository manages
        """
        self.session = session
        self.model_class = model_class
        self.logger = logging.getLogger(f"{__name__}.{model_class.__name__}")
    
    def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: Entity to create
            
        Returns:
            Created entity with generated ID
        """
        try:
            self.session.add(entity)
            self.session.flush()
            self.logger.debug(f"Created {self.model_class.__name__}: {entity.id}")
            return entity
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating {self.model_class.__name__}: {e}")
            self.session.rollback()
            raise
    
    def create_many(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in batch.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities
        """
        try:
            self.session.add_all(entities)
            self.session.flush()
            self.logger.debug(f"Created {len(entities)} {self.model_class.__name__} entities")
            return entities
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating batch {self.model_class.__name__}: {e}")
            self.session.rollback()
            raise
    
    def get_by_id(self, entity_id: UUID) -> Optional[T]:
        """
        Get entity by ID.
        
        Args:
            entity_id: UUID of entity
            
        Returns:
            Entity if found, None otherwise
        """
        try:
            return self.session.query(self.model_class).filter(
                self.model_class.id == entity_id
            ).first()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting {self.model_class.__name__} by ID: {e}")
            raise
    
    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = 'created_at',
        order_desc: bool = True
    ) -> List[T]:
        """
        Get all entities with pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Column to order by
            order_desc: Order descending if True
            
        Returns:
            List of entities
        """
        try:
            query = self.session.query(self.model_class)
            
            order_column = getattr(self.model_class, order_by, None)
            if order_column is not None:
                query = query.order_by(desc(order_column) if order_desc else asc(order_column))
            
            return query.offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error getting all {self.model_class.__name__}: {e}")
            raise
    
    def update(self, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity: Entity with updated values
            
        Returns:
            Updated entity
        """
        try:
            self.session.merge(entity)
            self.session.flush()
            self.logger.debug(f"Updated {self.model_class.__name__}: {entity.id}")
            return entity
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating {self.model_class.__name__}: {e}")
            self.session.rollback()
            raise
    
    def delete(self, entity_id: UUID) -> bool:
        """
        Delete entity by ID.
        
        Args:
            entity_id: UUID of entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            entity = self.get_by_id(entity_id)
            if entity:
                self.session.delete(entity)
                self.session.flush()
                self.logger.debug(f"Deleted {self.model_class.__name__}: {entity_id}")
                return True
            return False
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting {self.model_class.__name__}: {e}")
            self.session.rollback()
            raise
    
    def count(self, **filters) -> int:
        """
        Count entities matching filters.
        
        Args:
            **filters: Column=value pairs for filtering
            
        Returns:
            Count of matching entities
        """
        try:
            query = self.session.query(self.model_class)
            for key, value in filters.items():
                column = getattr(self.model_class, key, None)
                if column is not None:
                    query = query.filter(column == value)
            return query.count()
        except SQLAlchemyError as e:
            self.logger.error(f"Error counting {self.model_class.__name__}: {e}")
            raise
    
    def exists(self, entity_id: UUID) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: UUID to check
            
        Returns:
            True if exists
        """
        return self.get_by_id(entity_id) is not None
    
    def find_by(self, **filters) -> List[T]:
        """
        Find entities by column values.
        
        Args:
            **filters: Column=value pairs for filtering
            
        Returns:
            List of matching entities
        """
        try:
            query = self.session.query(self.model_class)
            for key, value in filters.items():
                column = getattr(self.model_class, key, None)
                if column is not None:
                    if isinstance(value, list):
                        query = query.filter(column.in_(value))
                    else:
                        query = query.filter(column == value)
            return query.all()
        except SQLAlchemyError as e:
            self.logger.error(f"Error finding {self.model_class.__name__}: {e}")
            raise
    
    def find_one_by(self, **filters) -> Optional[T]:
        """
        Find single entity by column values.
        
        Args:
            **filters: Column=value pairs for filtering
            
        Returns:
            Entity if found, None otherwise
        """
        results = self.find_by(**filters)
        return results[0] if results else None

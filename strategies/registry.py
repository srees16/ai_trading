"""
Strategy Registry Module.

Provides centralized registration and discovery of trading strategies.
Strategies are automatically registered when their modules are imported.
"""

from typing import Type, Optional
from .base_strategy import BaseStrategy, StrategyCategory
import logging

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Central registry for all trading strategies.
    
    This class provides a singleton-like registry that stores all available
    strategies. Strategies register themselves automatically upon import.
    
    Class Methods:
        register: Register a new strategy class
        get: Get a strategy class by name
        list_all: List all registered strategies
        list_by_category: List strategies filtered by category
        unregister: Remove a strategy from registry
        clear: Clear all registered strategies
    
    Example:
        ```python
        # Strategies auto-register via decorator
        @StrategyRegistry.register_decorator
        class MyStrategy(BaseStrategy):
            name = "My Strategy"
            ...
        
        # Or manually register
        StrategyRegistry.register(MyStrategy)
        
        # Get and use
        strategy_cls = StrategyRegistry.get("my_strategy")
        strategy = strategy_cls()
        results = strategy.run(...)
        ```
    """
    
    _strategies: dict[str, Type[BaseStrategy]] = {}
    _initialized: bool = False
    
    @classmethod
    def register(
        cls,
        strategy_class: Type[BaseStrategy],
        name: Optional[str] = None
    ) -> None:
        """
        Register a strategy class.
        
        Args:
            strategy_class: The strategy class to register (must inherit BaseStrategy)
            name: Optional override for the strategy name/key
        
        Raises:
            TypeError: If strategy_class doesn't inherit from BaseStrategy
            ValueError: If a strategy with the same name is already registered
        """
        # Validate inheritance
        if not isinstance(strategy_class, type) or not issubclass(strategy_class, BaseStrategy):
            raise TypeError(
                f"Strategy must inherit from BaseStrategy, got {type(strategy_class)}"
            )
        
        # Skip the base class itself
        if strategy_class is BaseStrategy:
            return
        
        # Generate registry key
        if name:
            key = cls._normalize_name(name)
        else:
            key = cls._normalize_name(strategy_class.name)
        
        # Check for duplicates
        if key in cls._strategies:
            existing = cls._strategies[key]
            if existing is not strategy_class:
                logger.warning(
                    f"Strategy '{key}' already registered as {existing.__name__}. "
                    f"Overwriting with {strategy_class.__name__}"
                )
        
        cls._strategies[key] = strategy_class
        logger.debug(f"Registered strategy: {key} -> {strategy_class.__name__}")
    
    @classmethod
    def register_decorator(cls, strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """
        Decorator for registering a strategy class.
        
        Usage:
            @StrategyRegistry.register_decorator
            class MyStrategy(BaseStrategy):
                ...
        """
        cls.register(strategy_class)
        return strategy_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseStrategy]]:
        """
        Get a strategy class by name.
        
        Args:
            name: Strategy name or key
        
        Returns:
            Strategy class or None if not found
        """
        key = cls._normalize_name(name)
        return cls._strategies.get(key)
    
    @classmethod
    def get_or_raise(cls, name: str) -> Type[BaseStrategy]:
        """
        Get a strategy class by name or raise error.
        
        Args:
            name: Strategy name or key
        
        Returns:
            Strategy class
        
        Raises:
            KeyError: If strategy not found
        """
        strategy = cls.get(name)
        if strategy is None:
            available = ", ".join(cls._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return strategy
    
    @classmethod
    def list_all(cls) -> dict[str, dict]:
        """
        List all registered strategies with their info.
        
        Returns:
            Dictionary mapping strategy keys to their info dictionaries
        """
        return {
            key: strategy_cls.get_info()
            for key, strategy_cls in cls._strategies.items()
        }
    
    @classmethod
    def list_by_category(cls, category: StrategyCategory) -> dict[str, dict]:
        """
        List strategies filtered by category.
        
        Args:
            category: StrategyCategory to filter by
        
        Returns:
            Dictionary of matching strategies
        """
        return {
            key: strategy_cls.get_info()
            for key, strategy_cls in cls._strategies.items()
            if strategy_cls.category == category
        }
    
    @classmethod
    def list_names(cls) -> list[str]:
        """
        Get list of all registered strategy names.
        
        Returns:
            List of strategy keys
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Remove a strategy from the registry.
        
        Args:
            name: Strategy name or key
        
        Returns:
            True if removed, False if not found
        """
        key = cls._normalize_name(name)
        if key in cls._strategies:
            del cls._strategies[key]
            logger.debug(f"Unregistered strategy: {key}")
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies."""
        cls._strategies.clear()
        cls._initialized = False
        logger.debug("Cleared strategy registry")
    
    @classmethod
    def count(cls) -> int:
        """Get count of registered strategies."""
        return len(cls._strategies)
    
    @classmethod
    def _normalize_name(cls, name: str) -> str:
        """
        Normalize strategy name for consistent lookup.
        
        Converts to lowercase and replaces spaces/hyphens with underscores.
        """
        return name.lower().replace(" ", "_").replace("-", "_")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if registry has been initialized with strategies."""
        return cls._initialized
    
    @classmethod
    def set_initialized(cls) -> None:
        """Mark registry as initialized."""
        cls._initialized = True


def register_strategy(name: Optional[str] = None):
    """
    Decorator factory for registering strategies with custom names.
    
    Args:
        name: Optional custom name for the strategy
    
    Usage:
        @register_strategy("custom_name")
        class MyStrategy(BaseStrategy):
            ...
    """
    def decorator(strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        StrategyRegistry.register(strategy_class, name)
        return strategy_class
    
    return decorator

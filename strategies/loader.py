"""
Dynamic Strategy Loader Module.

Provides automatic discovery and loading of strategy modules from
the trading_strategies directory. New strategies are automatically
detected and registered without modifying core code.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Type
import logging

from .base_strategy import BaseStrategy
from .registry import StrategyRegistry

logger = logging.getLogger(__name__)


# Module name mapping for known strategies
STRATEGY_MODULE_MAP = {
    # Momentum strategies
    "macd_oscillator": "trading_strategies.momentum_trading.macd_oscillator",
    "awesome_oscillator": "trading_strategies.momentum_trading.awesome_oscillator",
    "heikin_ashi": "trading_strategies.momentum_trading.heikin_ashi",
    "parabolic_sar": "trading_strategies.momentum_trading.parabolic_sar",
    
    # Pattern recognition strategies
    "rsi_pattern": "trading_strategies.pattern_recognition.rsi_pattern",
    "shooting_star": "trading_strategies.pattern_recognition.shooting_star",
    "support_resistance": "trading_strategies.pattern_recognition.support_resistance",
    
    # Statistical arbitrage strategies
    "pairs_trading": "trading_strategies.statistical_arbitrage.pairs_trading",
}


def get_trading_strategies_path() -> Path:
    """
    Get the path to the trading_strategies directory.
    
    Returns:
        Path object pointing to trading_strategies folder
    """
    # Start from this file's location
    current_dir = Path(__file__).parent
    
    # trading_strategies should be a sibling directory
    strategies_dir = current_dir.parent / "trading_strategies"
    
    if strategies_dir.exists():
        return strategies_dir
    
    # Fallback: check relative to working directory
    alt_path = Path.cwd() / "trading_strategies"
    if alt_path.exists():
        return alt_path
    
    return strategies_dir


def discover_strategies() -> dict[str, Path]:
    """
    Discover all strategy modules in the trading_strategies directory.
    
    Scans all subdirectories and finds Python files that could contain
    strategy implementations.
    
    Returns:
        Dictionary mapping strategy names to their file paths
    """
    strategies_dir = get_trading_strategies_path()
    discovered = {}
    
    if not strategies_dir.exists():
        logger.warning(f"Trading strategies directory not found: {strategies_dir}")
        return discovered
    
    logger.debug(f"Scanning for strategies in: {strategies_dir}")
    
    # Scan each category subfolder
    for category_dir in strategies_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        if category_dir.name.startswith(('_', '.')):
            continue
        
        # Find Python files
        for py_file in category_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            # Generate strategy key
            strategy_name = py_file.stem
            if strategy_name.endswith("_bktest"):
                strategy_name = strategy_name[:-7]  # Remove _bktest suffix
            
            key = f"{category_dir.name}/{strategy_name}"
            discovered[key] = py_file
            logger.debug(f"Discovered strategy: {key} -> {py_file}")
    
    return discovered


def load_strategy_module(
    module_path: Path | str,
    module_name: Optional[str] = None
):
    """
    Load a strategy module from a file path.
    
    Args:
        module_path: Path to the Python file
        module_name: Optional module name, auto-generated if not provided
    
    Returns:
        Loaded module object
    """
    if isinstance(module_path, str):
        module_path = Path(module_path)
    
    if not module_path.exists():
        raise FileNotFoundError(f"Strategy module not found: {module_path}")
    
    # Generate module name
    if module_name is None:
        module_name = f"trading_strategies.{module_path.parent.name}.{module_path.stem}"
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Error loading module {module_path}: {e}")
        raise
    
    return module


def find_strategy_classes(module) -> list[Type[BaseStrategy]]:
    """
    Find all BaseStrategy subclasses in a module.
    
    Args:
        module: Loaded Python module
    
    Returns:
        List of strategy classes found in the module
    """
    strategies = []
    
    for name in dir(module):
        obj = getattr(module, name)
        
        # Check if it's a class that inherits from BaseStrategy
        if (
            isinstance(obj, type) and
            issubclass(obj, BaseStrategy) and
            obj is not BaseStrategy
        ):
            strategies.append(obj)
    
    return strategies


def load_all_strategies(
    register: bool = True,
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None
) -> dict[str, Type[BaseStrategy]]:
    """
    Discover and load all strategies from trading_strategies directory.
    
    This is the main entry point for populating the strategy registry.
    It scans the trading_strategies folder, loads all Python modules,
    and extracts strategy classes.
    
    Args:
        register: Whether to register discovered strategies
        include_patterns: Optional list of patterns to include (e.g., ["momentum_*"])
        exclude_patterns: Optional list of patterns to exclude
    
    Returns:
        Dictionary mapping strategy names to their classes
    
    Example:
        ```python
        # Load and register all strategies
        from strategies import load_all_strategies
        
        strategies = load_all_strategies()
        print(f"Loaded {len(strategies)} strategies")
        ```
    """
    if StrategyRegistry.is_initialized():
        logger.debug("Registry already initialized, returning existing strategies")
        return {k: StrategyRegistry.get(k) for k in StrategyRegistry.list_names()}
    
    discovered = discover_strategies()
    loaded = {}
    
    for key, path in discovered.items():
        # Check include/exclude patterns
        if include_patterns:
            if not any(pattern in key for pattern in include_patterns):
                continue
        
        if exclude_patterns:
            if any(pattern in key for pattern in exclude_patterns):
                continue
        
        try:
            module = load_strategy_module(path)
            strategies = find_strategy_classes(module)
            
            for strategy_cls in strategies:
                strategy_key = strategy_cls.name.lower().replace(" ", "_")
                loaded[strategy_key] = strategy_cls
                
                if register:
                    StrategyRegistry.register(strategy_cls)
                
                logger.info(f"Loaded strategy: {strategy_cls.name}")
        
        except Exception as e:
            logger.warning(f"Failed to load strategy from {path}: {e}")
    
    if register:
        StrategyRegistry.set_initialized()
    
    logger.info(f"Loaded {len(loaded)} strategies")
    return loaded


def load_strategy_by_name(name: str) -> Optional[Type[BaseStrategy]]:
    """
    Load a specific strategy by name.
    
    Attempts to find and load a strategy module matching the name.
    
    Args:
        name: Strategy name or path
    
    Returns:
        Strategy class or None if not found
    """
    # First check registry
    existing = StrategyRegistry.get(name)
    if existing:
        return existing
    
    # Search discovered strategies
    discovered = discover_strategies()
    
    normalized_name = name.lower().replace(" ", "_").replace("-", "_")
    
    for key, path in discovered.items():
        if normalized_name in key.lower():
            try:
                module = load_strategy_module(path)
                strategies = find_strategy_classes(module)
                
                for strategy_cls in strategies:
                    StrategyRegistry.register(strategy_cls)
                    if strategy_cls.name.lower().replace(" ", "_") == normalized_name:
                        return strategy_cls
                
                if strategies:
                    return strategies[0]
            
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
    
    return None


def get_strategy_categories() -> dict[str, list[str]]:
    """
    Get all strategies organized by category.
    
    Returns:
        Dictionary mapping category names to lists of strategy names
    """
    if not StrategyRegistry.is_initialized():
        load_all_strategies()
    
    categories = {}
    
    for key, info in StrategyRegistry.list_all().items():
        category = info.get('category', 'other')
        if category not in categories:
            categories[category] = []
        categories[category].append(key)
    
    return categories


def reload_strategy(name: str) -> Optional[Type[BaseStrategy]]:
    """
    Reload a strategy module (useful for development).
    
    Args:
        name: Strategy name
    
    Returns:
        Reloaded strategy class
    """
    # Unregister existing
    StrategyRegistry.unregister(name)
    
    # Find and reload
    discovered = discover_strategies()
    normalized_name = name.lower().replace(" ", "_")
    
    for key, path in discovered.items():
        if normalized_name in key.lower():
            # Clear from sys.modules
            module_name = f"trading_strategies.{path.parent.name}.{path.stem}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Reload
            return load_strategy_by_name(name)
    
    return None

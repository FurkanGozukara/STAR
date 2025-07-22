"""
SeedVR2 Lazy Import Utilities

This module provides lazy import functionality for SeedVR2 components to reduce
startup time and memory usage. Modules are only imported when actually needed.

Key Features:
- Lazy loading of heavy SeedVR2 modules
- Automatic dependency checking and error handling
- Import caching to avoid repeated imports
- Thread-safe import management
- Graceful fallbacks when modules are unavailable

Performance Benefits:
- Reduces initial application startup time
- Lower memory usage when SeedVR2 is not used
- Better error isolation for missing dependencies
"""

import os
import sys
import threading
import logging
from typing import Optional, Dict, Any, Callable
from functools import lru_cache

# Thread lock for safe lazy imports
_import_lock = threading.Lock()

# Cache for imported modules to avoid repeated imports
_imported_modules: Dict[str, Any] = {}

# SeedVR2 availability status
_seedvr2_available: Optional[bool] = None


class LazyImportError(ImportError):
    """Custom exception for lazy import failures with helpful error messages."""
    pass


def get_seedvr2_base_path() -> str:
    """Get the SeedVR2 base directory path."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')


def ensure_seedvr2_path():
    """Ensure SeedVR2 is in Python path for imports."""
    seedvr2_path = get_seedvr2_base_path()
    if seedvr2_path not in sys.path:
        sys.path.insert(0, seedvr2_path)


@lru_cache(maxsize=1)
def check_seedvr2_availability() -> bool:
    """Check if SeedVR2 modules are available (cached result)."""
    global _seedvr2_available
    
    if _seedvr2_available is not None:
        return _seedvr2_available
    
    try:
        ensure_seedvr2_path()
        
        # Try importing core SeedVR2 modules
        import src.core.generation
        import src.core.model_manager
        import src.utils.downloads
        
        _seedvr2_available = True
        return True
        
    except ImportError:
        _seedvr2_available = False
        return False


def lazy_import_seedvr2_module(
    module_path: str, 
    logger: Optional[logging.Logger] = None,
    required: bool = True
) -> Any:
    """
    Lazily import a SeedVR2 module with caching and error handling.
    
    Args:
        module_path: Python module path (e.g., 'src.core.generation')
        logger: Optional logger for error reporting
        required: Whether the module is required (raises exception if True and import fails)
        
    Returns:
        Imported module or None if import fails and not required
        
    Raises:
        LazyImportError: If required=True and import fails
    """
    with _import_lock:
        # Check cache first
        if module_path in _imported_modules:
            return _imported_modules[module_path]
        
        # Check SeedVR2 availability
        if not check_seedvr2_availability():
            error_msg = f"SeedVR2 not available for module: {module_path}"
            if logger:
                logger.warning(error_msg)
            if required:
                raise LazyImportError(f"{error_msg}. Please ensure SeedVR2 is properly installed.")
            return None
        
        try:
            ensure_seedvr2_path()
            
            # Import the module
            module = __import__(module_path, fromlist=[''])
            
            # Cache the imported module
            _imported_modules[module_path] = module
            
            if logger:
                logger.debug(f"Successfully lazy-imported: {module_path}")
            
            return module
            
        except ImportError as e:
            error_msg = f"Failed to import SeedVR2 module '{module_path}': {e}"
            if logger:
                logger.error(error_msg)
            
            if required:
                raise LazyImportError(f"{error_msg}. Check SeedVR2 installation and dependencies.")
            
            return None


def lazy_import_seedvr2_function(
    module_path: str, 
    function_name: str,
    logger: Optional[logging.Logger] = None,
    required: bool = True
) -> Optional[Callable]:
    """
    Lazily import a specific function from a SeedVR2 module.
    
    Args:
        module_path: Python module path
        function_name: Name of function to import
        logger: Optional logger
        required: Whether the function is required
        
    Returns:
        Function object or None if import fails
    """
    module = lazy_import_seedvr2_module(module_path, logger, required)
    
    if module is None:
        return None
    
    try:
        func = getattr(module, function_name)
        if logger:
            logger.debug(f"Successfully imported function: {module_path}.{function_name}")
        return func
        
    except AttributeError:
        error_msg = f"Function '{function_name}' not found in module '{module_path}'"
        if logger:
            logger.error(error_msg)
        
        if required:
            raise LazyImportError(f"{error_msg}. Check SeedVR2 version compatibility.")
        
        return None


def get_lazy_generation_loop(logger: Optional[logging.Logger] = None) -> Optional[Callable]:
    """Get the SeedVR2 generation_loop function with lazy loading."""
    return lazy_import_seedvr2_function('src.core.generation', 'generation_loop', logger)


def get_lazy_configure_runner(logger: Optional[logging.Logger] = None) -> Optional[Callable]:
    """Get the SeedVR2 configure_runner function with lazy loading."""
    return lazy_import_seedvr2_function('src.core.model_manager', 'configure_runner', logger)


def get_lazy_download_weight(logger: Optional[logging.Logger] = None) -> Optional[Callable]:
    """Get the SeedVR2 download_weight function with lazy loading."""
    return lazy_import_seedvr2_function('src.utils.downloads', 'download_weight', logger)


def get_lazy_wavelet_reconstruction(logger: Optional[logging.Logger] = None) -> Optional[Callable]:
    """Get the SeedVR2 wavelet_reconstruction function with lazy loading."""
    return lazy_import_seedvr2_function('src.utils.color_fix', 'wavelet_reconstruction', logger)


def get_lazy_apply_block_swap_to_dit(logger: Optional[logging.Logger] = None) -> Optional[Callable]:
    """Get the SeedVR2 apply_block_swap_to_dit function with lazy loading."""
    return lazy_import_seedvr2_function('src.optimization.blockswap', 'apply_block_swap_to_dit', logger)


def clear_import_cache():
    """Clear the imported modules cache. Useful for testing or reloading modules."""
    global _imported_modules, _seedvr2_available
    with _import_lock:
        _imported_modules.clear()
        _seedvr2_available = None
        # Clear the lru_cache for check_seedvr2_availability
        check_seedvr2_availability.cache_clear()


def get_import_stats() -> Dict[str, Any]:
    """Get statistics about lazy imports for debugging."""
    return {
        "seedvr2_available": _seedvr2_available,
        "cached_modules": list(_imported_modules.keys()),
        "cache_size": len(_imported_modules),
        "seedvr2_path": get_seedvr2_base_path(),
        "path_in_sys_path": get_seedvr2_base_path() in sys.path
    }


# Context manager for temporary import path management
class SeedVR2ImportContext:
    """Context manager for safe SeedVR2 import path management."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.path_added = False
        self.seedvr2_path = get_seedvr2_base_path()
    
    def __enter__(self):
        if self.seedvr2_path not in sys.path:
            sys.path.insert(0, self.seedvr2_path)
            self.path_added = True
            if self.logger:
                self.logger.debug(f"Added SeedVR2 path to sys.path: {self.seedvr2_path}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path_added and self.seedvr2_path in sys.path:
            try:
                sys.path.remove(self.seedvr2_path)
                if self.logger:
                    self.logger.debug(f"Removed SeedVR2 path from sys.path: {self.seedvr2_path}")
            except ValueError:
                # Path was already removed, ignore
                pass 
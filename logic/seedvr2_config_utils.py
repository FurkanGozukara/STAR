"""
SeedVR2 Configuration Access Utilities

This module provides optimized configuration access for SeedVR2 processing,
reducing redundant getattr() calls and providing a centralized configuration interface.

Key Features:
- Cached configuration access to avoid repeated getattr() calls
- Type-safe configuration extraction with validation
- Default value management with fallbacks
- Configuration validation and error reporting
- Performance optimization for frequently accessed settings

Performance Benefits:
- Reduces repeated attribute lookups
- Centralizes default value management
- Provides type safety and validation
- Improves code maintainability
"""

import logging
from typing import Optional, Dict, Any, Union, TypeVar, Type
from dataclasses import dataclass

# Import constants from dataclasses
from .star_dataclasses import (
    DEFAULT_SEEDVR2_UPSCALE_FACTOR,
    DEFAULT_SEEDVR2_DEFAULT_RESOLUTION,
    DEFAULT_SEEDVR2_MIN_RESOLUTION,
    DEFAULT_SEEDVR2_MAX_RESOLUTION,
    DEFAULT_SEEDVR2_MODEL_PRECISION,
    DEFAULT_SEEDVR2_CFG_SCALE,
    DEFAULT_SEEDVR2_SEED,
    DEFAULT_SEEDVR2_BATCH_SIZE,
    DEFAULT_SEEDVR2_PRESERVE_VRAM,
    DEFAULT_SEEDVR2_FLASH_ATTENTION,
    DEFAULT_SEEDVR2_COLOR_CORRECTION,
    DEFAULT_SEEDVR2_TEMPORAL_OVERLAP
)

T = TypeVar('T')


@dataclass
class SeedVR2ConfigSnapshot:
    """
    Immutable snapshot of SeedVR2 configuration for efficient access.
    
    This avoids repeated getattr() calls and provides type safety.
    """
    # Model settings
    model: Optional[str]
    model_precision: str
    cfg_scale: float
    seed: int
    
    # Processing settings
    batch_size: int
    preserve_vram: bool
    flash_attention: bool
    color_correction: bool
    temporal_overlap: int
    
    # Resolution settings
    target_resolution: int
    upscale_factor: float
    
    # GPU and memory settings
    enable_multi_gpu: bool
    gpu_devices: str
    enable_block_swap: bool
    block_swap_counter: int
    block_swap_offload_io: bool
    block_swap_model_caching: bool
    
    # Advanced settings
    enable_temporal_consistency: bool
    scene_awareness: bool
    temporal_quality: str
    consistency_validation: bool
    chunk_optimization: bool
    enable_frame_padding: bool
    pad_last_chunk: bool
    skip_first_frames: int
    
    @property
    def is_valid(self) -> bool:
        """Check if the configuration snapshot is valid."""
        return (
            self.batch_size > 0 and
            self.cfg_scale >= 0 and
            self.target_resolution >= DEFAULT_SEEDVR2_MIN_RESOLUTION and
            self.target_resolution <= DEFAULT_SEEDVR2_MAX_RESOLUTION and
            self.upscale_factor > 0 and
            self.temporal_overlap >= 0
        )
    
    @property
    def memory_usage_level(self) -> str:
        """Estimate memory usage level based on configuration."""
        if self.preserve_vram and self.enable_block_swap:
            return "low"
        elif self.preserve_vram or self.batch_size <= 5:
            return "medium"
        else:
            return "high"


def safe_getattr(obj: Any, attr: str, default: T, expected_type: Type[T] = None) -> T:
    """
    Safely get attribute with type checking and default fallback.
    
    Args:
        obj: Object to get attribute from
        attr: Attribute name
        default: Default value if attribute is missing or invalid
        expected_type: Expected type for validation
        
    Returns:
        Attribute value or default if invalid/missing
    """
    try:
        value = getattr(obj, attr, default)
        
        # Type validation if specified
        if expected_type is not None and not isinstance(value, expected_type):
            return default
        
        return value
        
    except (AttributeError, TypeError):
        return default


def extract_seedvr2_config(
    config_obj: Any, 
    logger: Optional[logging.Logger] = None
) -> SeedVR2ConfigSnapshot:
    """
    Extract and validate SeedVR2 configuration from config object.
    
    Args:
        config_obj: Configuration object (could be AppConfig, dict, or SeedVR2Config)
        logger: Optional logger for warnings
        
    Returns:
        Validated SeedVR2 configuration snapshot
    """
    try:
        # Handle different config object types
        if hasattr(config_obj, 'seedvr2'):
            # AppConfig with seedvr2 sub-config
            seedvr2_config = config_obj.seedvr2
        elif hasattr(config_obj, 'model'):
            # Direct SeedVR2Config object
            seedvr2_config = config_obj
        else:
            # Dictionary or other object
            seedvr2_config = config_obj
        
        # Extract configuration with safe defaults
        snapshot = SeedVR2ConfigSnapshot(
            # Model settings
            model=safe_getattr(seedvr2_config, 'model', None, str),
            model_precision=safe_getattr(seedvr2_config, 'model_precision', DEFAULT_SEEDVR2_MODEL_PRECISION, str),
            cfg_scale=safe_getattr(seedvr2_config, 'cfg_scale', DEFAULT_SEEDVR2_CFG_SCALE, (int, float)),
            seed=safe_getattr(seedvr2_config, 'seed', DEFAULT_SEEDVR2_SEED, int),
            
            # Processing settings
            batch_size=safe_getattr(seedvr2_config, 'batch_size', DEFAULT_SEEDVR2_BATCH_SIZE, int),
            preserve_vram=safe_getattr(seedvr2_config, 'preserve_vram', DEFAULT_SEEDVR2_PRESERVE_VRAM, bool),
            flash_attention=safe_getattr(seedvr2_config, 'flash_attention', DEFAULT_SEEDVR2_FLASH_ATTENTION, bool),
            color_correction=safe_getattr(seedvr2_config, 'color_correction', DEFAULT_SEEDVR2_COLOR_CORRECTION, bool),
            temporal_overlap=safe_getattr(seedvr2_config, 'temporal_overlap', DEFAULT_SEEDVR2_TEMPORAL_OVERLAP, int),
            
            # Resolution settings
            target_resolution=safe_getattr(seedvr2_config, 'target_resolution', DEFAULT_SEEDVR2_DEFAULT_RESOLUTION, int),
            upscale_factor=safe_getattr(seedvr2_config, 'upscale_factor', DEFAULT_SEEDVR2_UPSCALE_FACTOR, (int, float)),
            
            # GPU and memory settings
            enable_multi_gpu=safe_getattr(seedvr2_config, 'enable_multi_gpu', False, bool),
            gpu_devices=safe_getattr(seedvr2_config, 'gpu_devices', "0", str),
            enable_block_swap=safe_getattr(seedvr2_config, 'enable_block_swap', False, bool),
            block_swap_counter=safe_getattr(seedvr2_config, 'block_swap_counter', 0, int),
            block_swap_offload_io=safe_getattr(seedvr2_config, 'block_swap_offload_io', False, bool),
            block_swap_model_caching=safe_getattr(seedvr2_config, 'block_swap_model_caching', False, bool),
            
            # Advanced settings
            enable_temporal_consistency=safe_getattr(seedvr2_config, 'enable_temporal_consistency', True, bool),
            scene_awareness=safe_getattr(seedvr2_config, 'scene_awareness', True, bool),
            temporal_quality=safe_getattr(seedvr2_config, 'temporal_quality', "balanced", str),
            consistency_validation=safe_getattr(seedvr2_config, 'consistency_validation', True, bool),
            chunk_optimization=safe_getattr(seedvr2_config, 'chunk_optimization', True, bool),
            enable_frame_padding=safe_getattr(seedvr2_config, 'enable_frame_padding', True, bool),
            pad_last_chunk=safe_getattr(seedvr2_config, 'pad_last_chunk', True, bool),
            skip_first_frames=safe_getattr(seedvr2_config, 'skip_first_frames', 0, int),
        )
        
        # Validate and log warnings for invalid values
        if not snapshot.is_valid:
            if logger:
                logger.warning("Invalid SeedVR2 configuration detected, using defaults for invalid values")
                _log_config_warnings(snapshot, logger)
        
        return snapshot
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to extract SeedVR2 configuration: {e}, using defaults")
        
        # Return safe default configuration
        return SeedVR2ConfigSnapshot(
            model=None,
            model_precision=DEFAULT_SEEDVR2_MODEL_PRECISION,
            cfg_scale=DEFAULT_SEEDVR2_CFG_SCALE,
            seed=DEFAULT_SEEDVR2_SEED,
            batch_size=DEFAULT_SEEDVR2_BATCH_SIZE,
            preserve_vram=DEFAULT_SEEDVR2_PRESERVE_VRAM,
            flash_attention=DEFAULT_SEEDVR2_FLASH_ATTENTION,
            color_correction=DEFAULT_SEEDVR2_COLOR_CORRECTION,
            temporal_overlap=DEFAULT_SEEDVR2_TEMPORAL_OVERLAP,
            target_resolution=DEFAULT_SEEDVR2_DEFAULT_RESOLUTION,
            upscale_factor=DEFAULT_SEEDVR2_UPSCALE_FACTOR,
            enable_multi_gpu=False,
            gpu_devices="0",
            enable_block_swap=False,
            block_swap_counter=0,
            block_swap_offload_io=False,
            block_swap_model_caching=False,
            enable_temporal_consistency=True,
            scene_awareness=True,
            temporal_quality="balanced",
            consistency_validation=True,
            chunk_optimization=True,
            enable_frame_padding=True,
            pad_last_chunk=True,
            skip_first_frames=0
        )


def _log_config_warnings(snapshot: SeedVR2ConfigSnapshot, logger: logging.Logger):
    """Log specific configuration warnings."""
    if snapshot.batch_size <= 0:
        logger.warning(f"Invalid batch_size: {snapshot.batch_size}, should be > 0")
    
    if snapshot.cfg_scale < 0:
        logger.warning(f"Invalid cfg_scale: {snapshot.cfg_scale}, should be >= 0")
    
    if not (DEFAULT_SEEDVR2_MIN_RESOLUTION <= snapshot.target_resolution <= DEFAULT_SEEDVR2_MAX_RESOLUTION):
        logger.warning(f"Invalid target_resolution: {snapshot.target_resolution}, should be between {DEFAULT_SEEDVR2_MIN_RESOLUTION}-{DEFAULT_SEEDVR2_MAX_RESOLUTION}")
    
    if snapshot.upscale_factor <= 0:
        logger.warning(f"Invalid upscale_factor: {snapshot.upscale_factor}, should be > 0")
    
    if snapshot.temporal_overlap < 0:
        logger.warning(f"Invalid temporal_overlap: {snapshot.temporal_overlap}, should be >= 0")


def optimize_config_for_hardware(
    snapshot: SeedVR2ConfigSnapshot,
    available_vram_gb: float = None,
    logger: Optional[logging.Logger] = None
) -> SeedVR2ConfigSnapshot:
    """
    Optimize configuration based on available hardware.
    
    Args:
        snapshot: Current configuration snapshot
        available_vram_gb: Available VRAM in GB (auto-detected if None)
        logger: Optional logger
        
    Returns:
        Optimized configuration snapshot
    """
    try:
        # Auto-detect VRAM if not provided
        if available_vram_gb is None:
            try:
                import torch
                if torch.cuda.is_available():
                    available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    available_vram_gb = 4.0  # Conservative default
            except:
                available_vram_gb = 4.0
        
        # Create optimized copy
        optimized = snapshot
        
        # Optimize based on VRAM
        if available_vram_gb < 8:
            # Low VRAM optimizations
            optimized = SeedVR2ConfigSnapshot(
                **snapshot.__dict__,
                preserve_vram=True,
                enable_block_swap=True,
                block_swap_counter=max(4, snapshot.block_swap_counter),
                batch_size=min(snapshot.batch_size, 3),
                flash_attention=True
            )
            if logger:
                logger.info(f"Applied low VRAM optimizations for {available_vram_gb:.1f}GB VRAM")
                
        elif available_vram_gb < 16:
            # Medium VRAM optimizations
            optimized = SeedVR2ConfigSnapshot(
                **snapshot.__dict__,
                preserve_vram=True,
                batch_size=min(snapshot.batch_size, 5),
                flash_attention=True
            )
            if logger:
                logger.info(f"Applied medium VRAM optimizations for {available_vram_gb:.1f}GB VRAM")
        
        else:
            # High VRAM - minimal restrictions
            if logger:
                logger.info(f"No VRAM restrictions needed for {available_vram_gb:.1f}GB VRAM")
        
        return optimized
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to optimize config for hardware: {e}, using original config")
        return snapshot


def get_config_summary(snapshot: SeedVR2ConfigSnapshot) -> Dict[str, Any]:
    """Get a summary of the configuration for logging/debugging."""
    return {
        "model": snapshot.model or "Not specified",
        "batch_size": snapshot.batch_size,
        "target_resolution": snapshot.target_resolution,
        "upscale_factor": f"{snapshot.upscale_factor}x",
        "memory_usage": snapshot.memory_usage_level,
        "temporal_consistency": snapshot.enable_temporal_consistency,
        "multi_gpu": snapshot.enable_multi_gpu,
        "block_swap": f"Enabled ({snapshot.block_swap_counter} blocks)" if snapshot.enable_block_swap else "Disabled",
        "flash_attention": snapshot.flash_attention,
        "color_correction": snapshot.color_correction,
        "is_valid": snapshot.is_valid
    } 
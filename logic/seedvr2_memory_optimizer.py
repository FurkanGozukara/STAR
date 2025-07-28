"""
SeedVR2 Intelligent Memory Optimization

This module provides intelligent memory optimization for SeedVR2 that automatically
configures settings based on GPU capabilities and user preferences.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch

from .star_dataclasses import SeedVR2Config

logger = logging.getLogger(__name__)


def get_memory_optimization_presets() -> Dict[str, Dict[str, Any]]:
    """
    Get memory optimization presets for user selection.
    
    Returns:
        Dict of preset name to configuration
    """
    return {
        "auto": {
            "description": "Automatically optimize based on GPU",
            "config": None  # Will be determined dynamically
        },
        "performance": {
            "description": "Maximum performance, high VRAM usage",
            "config": {
                "enable_block_swap": False,
                "block_swap_counter": 0,
                "preserve_vram": False,
                "memory_reserved_threshold": 8.0,
                "memory_fraction_low_reserved": 0.95,
                "memory_fraction_high_reserved": 0.9,
                "block_memory_cleanup_threshold": 0.9,
                "io_memory_cleanup_threshold": 0.95
            }
        },
        "balanced": {
            "description": "Balance between speed and memory",
            "config": {
                "enable_block_swap": True,
                "block_swap_counter": 8,
                "preserve_vram": True,
                "memory_reserved_threshold": 4.0,
                "memory_fraction_low_reserved": 0.8,
                "memory_fraction_high_reserved": 0.6,
                "block_memory_cleanup_threshold": 0.7,
                "io_memory_cleanup_threshold": 0.85
            }
        },
        "memory_saver": {
            "description": "Minimum VRAM usage, slower processing",
            "config": {
                "enable_block_swap": True,
                "block_swap_counter": 20,
                "block_swap_offload_io": True,
                "preserve_vram": True,
                "memory_reserved_threshold": 2.0,
                "memory_fraction_low_reserved": 0.6,
                "memory_fraction_high_reserved": 0.4,
                "block_memory_cleanup_threshold": 0.5,
                "io_memory_cleanup_threshold": 0.7
            }
        }
    }


def detect_gpu_capabilities() -> Dict[str, Any]:
    """
    Detect GPU capabilities and return optimization parameters.
    
    Returns:
        Dict with GPU info and recommended settings
    """
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "name": "No GPU",
            "vram_gb": 0,
            "compute_capability": (0, 0),
            "profile": "cpu"
        }
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    
    # Determine GPU profile
    if vram_gb >= 24:
        profile = "high_end"
    elif vram_gb >= 12:
        profile = "mid_range"  
    elif vram_gb >= 8:
        profile = "entry_level"
    else:
        profile = "low_vram"
    
    return {
        "gpu_available": True,
        "name": props.name,
        "vram_gb": vram_gb,
        "compute_capability": (props.major, props.minor),
        "profile": profile,
        "supports_fp8": props.major >= 9,  # Ada Lovelace and newer
        "supports_flash_attention": props.major >= 8  # Ampere and newer
    }


def apply_memory_optimization(config: Any, optimization_mode: str = "auto") -> Dict[str, Any]:
    """
    Apply memory optimization to SeedVR2 configuration.
    
    Args:
        config: AppConfig or similar configuration object
        optimization_mode: One of "auto", "performance", "balanced", "memory_saver"
        
    Returns:
        Dict with applied settings and recommendations
    """
    presets = get_memory_optimization_presets()
    gpu_info = detect_gpu_capabilities()
    
    # Get SeedVR2 config section
    seedvr2_config = getattr(config, 'seedvr2', None)
    if not seedvr2_config:
        logger.warning("No SeedVR2 configuration found")
        return {"status": "error", "message": "No SeedVR2 configuration"}
    
    # Get current model info
    model_name = getattr(seedvr2_config, 'model', '')
    is_7b_model = '7b' in model_name.lower()
    batch_size = getattr(seedvr2_config, 'batch_size', 5)
    
    if optimization_mode == "auto":
        # Intelligent auto-configuration based on GPU and model
        config_updates = _get_auto_config(gpu_info, is_7b_model, batch_size)
    else:
        # Use preset configuration
        preset = presets.get(optimization_mode)
        if not preset:
            logger.warning(f"Unknown optimization mode: {optimization_mode}")
            return {"status": "error", "message": f"Unknown mode: {optimization_mode}"}
        config_updates = preset["config"]
    
    # Apply configuration updates
    applied_settings = {}
    for key, value in config_updates.items():
        if hasattr(seedvr2_config, key):
            setattr(seedvr2_config, key, value)
            applied_settings[key] = value
    
    # Generate status message
    status_message = _generate_optimization_status(
        gpu_info, optimization_mode, applied_settings, is_7b_model
    )
    
    return {
        "status": "success",
        "gpu_info": gpu_info,
        "optimization_mode": optimization_mode,
        "applied_settings": applied_settings,
        "message": status_message,
        "recommendations": _get_recommendations(gpu_info, is_7b_model, optimization_mode)
    }


def _get_auto_config(gpu_info: Dict[str, Any], is_7b_model: bool, batch_size: int) -> Dict[str, Any]:
    """Get automatic configuration based on GPU and model."""
    vram_gb = gpu_info["vram_gb"]
    profile = gpu_info["profile"]
    
    # Base memory requirements
    base_vram_needed = 14.0 if is_7b_model else 8.0
    
    # Adjust for batch size
    if batch_size > 5:
        base_vram_needed += (batch_size - 5) * 0.5
    
    # Calculate how much block swap is needed
    vram_deficit = max(0, base_vram_needed - vram_gb * 0.8)  # Keep 20% buffer
    
    if profile == "high_end" and vram_deficit <= 0:
        # High-end GPU with enough VRAM
        return {
            "enable_block_swap": False,
            "block_swap_counter": 0,
            "preserve_vram": False,
            "memory_reserved_threshold": 8.0,
            "memory_fraction_low_reserved": 0.9,
            "memory_fraction_high_reserved": 0.85,
            "block_memory_cleanup_threshold": 0.85,
            "io_memory_cleanup_threshold": 0.95
        }
    
    elif profile == "mid_range" or (profile == "high_end" and vram_deficit > 0):
        # Need some optimization
        blocks_to_swap = min(16, int(vram_deficit * 2))
        return {
            "enable_block_swap": blocks_to_swap > 0,
            "block_swap_counter": blocks_to_swap,
            "preserve_vram": True,
            "memory_reserved_threshold": 4.0,
            "memory_fraction_low_reserved": 0.8,
            "memory_fraction_high_reserved": 0.6,
            "block_memory_cleanup_threshold": 0.7,
            "io_memory_cleanup_threshold": 0.85
        }
    
    else:  # entry_level or low_vram
        # Need aggressive optimization
        blocks_to_swap = min(28, max(16, int(vram_deficit * 3)))
        return {
            "enable_block_swap": True,
            "block_swap_counter": blocks_to_swap,
            "block_swap_offload_io": vram_gb < 8,
            "preserve_vram": True,
            "memory_reserved_threshold": 2.0,
            "memory_fraction_low_reserved": 0.6,
            "memory_fraction_high_reserved": 0.4,
            "block_memory_cleanup_threshold": 0.5,
            "io_memory_cleanup_threshold": 0.7
        }


def _generate_optimization_status(gpu_info: Dict, mode: str, settings: Dict, 
                                  is_7b_model: bool) -> str:
    """Generate user-friendly status message."""
    gpu_name = gpu_info.get("name", "Unknown GPU")
    vram_gb = gpu_info.get("vram_gb", 0)
    
    if mode == "auto":
        mode_desc = "Automatic optimization"
    else:
        mode_desc = get_memory_optimization_presets()[mode]["description"]
    
    blocks_swap = settings.get("block_swap_counter", 0)
    model_size = "7B" if is_7b_model else "3B"
    
    if blocks_swap == 0:
        swap_status = "No block swapping needed"
    else:
        swap_status = f"Swapping {blocks_swap} blocks to save ~{blocks_swap * 0.5:.1f}GB VRAM"
    
    return f"""
🎯 Memory Optimization: {mode_desc}
🖥️ GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)
🤖 Model: SeedVR2 {model_size}
💾 {swap_status}
"""


def _get_recommendations(gpu_info: Dict, is_7b_model: bool, mode: str) -> list:
    """Get optimization recommendations for the user."""
    recommendations = []
    vram_gb = gpu_info.get("vram_gb", 0)
    
    if is_7b_model and vram_gb < 16:
        recommendations.append("Consider using 3B model for better performance on your GPU")
    
    if mode == "performance" and vram_gb < 12:
        recommendations.append("Your GPU may not have enough VRAM for performance mode")
    
    if gpu_info.get("supports_flash_attention", False):
        recommendations.append("Flash Attention is supported and enabled for faster processing")
    
    if mode == "memory_saver":
        recommendations.append("Processing will be slower but use minimal VRAM")
    
    return recommendations
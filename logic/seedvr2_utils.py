"""
SeedVR2 Utility Functions for STAR Application

This module provides utility functions for SeedVR2 integration with STAR's UI and infrastructure.
It bridges the gap between SeedVR2 functionality and STAR's existing systems.

Key Features:
- Dependency checking and validation
- Model discovery and management
- UI helper functions
- Integration utilities
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any

# Import STAR utilities
from .seedvr2_model_manager import get_seedvr2_model_manager, check_seedvr2_dependencies, SEEDVR2_AVAILABLE

def util_check_seedvr2_dependencies(logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Check SeedVR2 dependencies for UI display
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Tuple of (all_available, missing_dependencies_list)
    """
    return check_seedvr2_dependencies(logger)

def util_scan_seedvr2_models(logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Scan for available SeedVR2 models
    
    Args:
        logger: Optional logger instance
        
    Returns:
        List of model dictionaries
    """
    if not SEEDVR2_AVAILABLE:
        return []
    
    model_manager = get_seedvr2_model_manager(logger)
    return model_manager.scan_available_models()

def util_get_seedvr2_model_info(model_name: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get information about a specific SeedVR2 model
    
    Args:
        model_name: Name of the model
        logger: Optional logger instance
        
    Returns:
        Dictionary with model information
    """
    if not SEEDVR2_AVAILABLE:
        return {"error": "SeedVR2 not available"}
    
    model_manager = get_seedvr2_model_manager(logger)
    return model_manager.get_model_info(model_name)

def util_get_vram_info(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get current VRAM information
    
    Args:
        logger: Optional logger instance
        
    Returns:
        Dictionary with VRAM information
    """
    if not SEEDVR2_AVAILABLE:
        return {"error": "SeedVR2 not available"}
    
    model_manager = get_seedvr2_model_manager(logger)
    return model_manager.get_vram_info()

def util_get_block_swap_recommendations(model_name: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get intelligent block swap recommendations for a model
    
    Args:
        model_name: Name of the model
        logger: Optional logger instance
        
    Returns:
        Dictionary with recommended settings
    """
    if not SEEDVR2_AVAILABLE:
        return {
            'blocks_to_swap': 16,
            'offload_io_components': False,
            'cache_model': False,
            'reason': 'SeedVR2 not available, using defaults'
        }
    
    model_manager = get_seedvr2_model_manager(logger)
    return model_manager.get_optimal_block_swap_recommendations(model_name)

def util_format_model_display_name(model_info: Dict[str, Any]) -> str:
    """
    Format a model's display name for UI dropdowns
    
    Args:
        model_info: Model information dictionary
        
    Returns:
        Formatted display name
    """
    if 'display_name' in model_info:
        return model_info['display_name']
    
    # Fallback formatting
    name = model_info.get('filename', 'Unknown Model')
    if not model_info.get('available', False):
        name += " (Download Required)"
    
    return name

def util_extract_model_filename_from_dropdown(dropdown_choice: str, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Extract the actual model filename from a dropdown choice
    
    Args:
        dropdown_choice: The display name from the dropdown
        logger: Optional logger instance
        
    Returns:
        The actual model filename or None if not found
    """
    if not dropdown_choice or "No SeedVR2" in dropdown_choice or "Error" in dropdown_choice:
        return None
    
    try:
        # Get all available models
        available_models = util_scan_seedvr2_models(logger)
        
        # Try to match by display name
        for model in available_models:
            display_name = util_format_model_display_name(model)
            if display_name == dropdown_choice:
                return model.get('filename')
        
        # If no exact match, try to find filename in the choice string
        # Handle cases like "SeedVR2 3B (FP8) - Enhanced sharpness variant (Download Required)"
        for model in available_models:
            filename = model.get('filename', '')
            if filename and (filename in dropdown_choice or 
                           any(part in dropdown_choice for part in filename.split('_') if len(part) > 2)):
                return filename
        
        # Fallback: if it looks like a direct filename, return it
        if dropdown_choice.endswith('.safetensors'):
            return dropdown_choice
            
        if logger:
            logger.warning(f"Could not extract filename from dropdown choice: {dropdown_choice}")
        
        return None
        
    except Exception as e:
        if logger:
            logger.error(f"Error extracting model filename: {e}")
        return None

def util_format_vram_status(vram_info: Dict[str, Any]) -> str:
    """
    Format VRAM information for UI display
    
    Args:
        vram_info: VRAM information dictionary
        
    Returns:
        Formatted VRAM status string
    """
    if 'error' in vram_info:
        return f"❌ {vram_info['error']}"
    
    free_gb = vram_info.get('free_gb', 0)
    total_gb = vram_info.get('total_gb', 0)
    
    if total_gb > 0:
        used_gb = total_gb - free_gb
        usage_percent = (used_gb / total_gb) * 100
        
        status_icon = "🟢" if usage_percent < 70 else "🟡" if usage_percent < 90 else "🔴"
        
        return f"{status_icon} VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({usage_percent:.1f}% used)"
    else:
        return "❌ VRAM information unavailable"

def util_format_block_swap_status(recommendations: Dict[str, Any]) -> str:
    """
    Format block swap recommendations for UI display
    
    Args:
        recommendations: Block swap recommendations dictionary
        
    Returns:
        Formatted status string
    """
    blocks = recommendations.get('blocks_to_swap', 0)
    reason = recommendations.get('reason', 'No information available')
    
    if blocks == 0:
        icon = "🟢"
        status = "Block Swap: Disabled (Sufficient VRAM)"
    elif blocks <= 8:
        icon = "🟡"
        status = f"Block Swap: Light ({blocks} blocks)"
    elif blocks <= 16:
        icon = "🟠"
        status = f"Block Swap: Moderate ({blocks} blocks)"
    else:
        icon = "🔴"
        status = f"Block Swap: Aggressive ({blocks} blocks)"
    
    return f"{icon} {status}\n💡 {reason}"

def util_validate_seedvr2_config(config_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate SeedVR2 configuration parameters
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check model selection
    model_name = config_dict.get('seedvr2_model')
    if not model_name:
        errors.append("No SeedVR2 model selected")
    
    # Check batch size for temporal consistency
    batch_size = config_dict.get('seedvr2_batch_size', 5)
    if batch_size < 1:
        errors.append("Batch size must be at least 1")
    elif batch_size < 5:
        errors.append("Warning: Batch size < 5 disables temporal consistency")
    
    # Check resolution
    resolution = config_dict.get('seedvr2_resolution', 1072)
    if resolution < 16 or resolution > 4320:
        errors.append("Resolution must be between 16 and 4320")
    elif resolution % 16 != 0:
        errors.append("Resolution should be divisible by 16 for optimal performance")
    
    # Check block swap settings
    blocks_to_swap = config_dict.get('seedvr2_blocks_to_swap', 0)
    if blocks_to_swap < 0 or blocks_to_swap > 36:
        errors.append("Blocks to swap must be between 0 and 36")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def util_get_suggested_settings(model_name: str, vram_gb: float) -> Dict[str, Any]:
    """
    Get suggested settings based on model and available VRAM
    
    Args:
        model_name: Name of the model
        vram_gb: Available VRAM in GB
        
    Returns:
        Dictionary with suggested settings
    """
    suggestions = {
        'batch_size': 5,  # Minimum for temporal consistency
        'blocks_to_swap': 0,
        'offload_io_components': False,
        'cache_model': True,
        'preserve_vram': True
    }
    
    # Adjust based on model size
    if '7b' in model_name.lower():
        # 7B model needs more VRAM
        if vram_gb < 16:
            suggestions['batch_size'] = 1
            suggestions['blocks_to_swap'] = 24
            suggestions['offload_io_components'] = True
            suggestions['cache_model'] = False
        elif vram_gb < 20:
            suggestions['batch_size'] = 5
            suggestions['blocks_to_swap'] = 16
            suggestions['offload_io_components'] = True
        elif vram_gb < 24:
            suggestions['batch_size'] = 8
            suggestions['blocks_to_swap'] = 8
    else:
        # 3B model is more VRAM efficient
        if vram_gb < 12:
            suggestions['batch_size'] = 1
            suggestions['blocks_to_swap'] = 16
            suggestions['offload_io_components'] = True
            suggestions['cache_model'] = False
        elif vram_gb < 16:
            suggestions['batch_size'] = 5
            suggestions['blocks_to_swap'] = 8
        elif vram_gb >= 18:
            suggestions['batch_size'] = 12
    
    # FP8 models use less VRAM
    if 'fp8' in model_name.lower():
        suggestions['batch_size'] = min(suggestions['batch_size'] * 2, 16)
        suggestions['blocks_to_swap'] = max(0, suggestions['blocks_to_swap'] - 8)
    
    return suggestions

def util_estimate_processing_time(video_frames: int, batch_size: int, model_name: str) -> Dict[str, Any]:
    """
    Estimate processing time for video upscaling
    
    Args:
        video_frames: Number of frames in video
        batch_size: Batch size for processing
        model_name: Name of the model
        
    Returns:
        Dictionary with time estimates
    """
    # Base processing time per frame (in seconds) - rough estimates
    base_time_per_frame = 2.0  # Base time for 3B model
    
    if '7b' in model_name.lower():
        base_time_per_frame *= 1.8  # 7B models are slower
    
    if 'fp8' in model_name.lower():
        base_time_per_frame *= 0.7  # FP8 is faster
    
    # Batch efficiency (larger batches are more efficient per frame)
    batch_efficiency = min(1.0, 0.5 + (batch_size / 20))
    effective_time_per_frame = base_time_per_frame * batch_efficiency
    
    total_seconds = video_frames * effective_time_per_frame
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    # Format time estimate
    if total_hours >= 1:
        time_str = f"{total_hours:.1f} hours"
    elif total_minutes >= 1:
        time_str = f"{total_minutes:.1f} minutes"
    else:
        time_str = f"{total_seconds:.0f} seconds"
    
    return {
        'total_seconds': total_seconds,
        'total_minutes': total_minutes,
        'total_hours': total_hours,
        'formatted_time': time_str,
        'frames_per_batch': batch_size,
        'total_batches': (video_frames + batch_size - 1) // batch_size
    }

def util_cleanup_seedvr2_resources(logger: Optional[logging.Logger] = None):
    """
    Cleanup all SeedVR2 resources and free memory
    
    Args:
        logger: Optional logger instance
    """
    if not SEEDVR2_AVAILABLE:
        return
    
    try:
        # Cleanup inference engine (import locally to avoid circular deps)
        try:
            from .seedvr2_inference import get_seedvr2_inference_engine
            inference_engine = get_seedvr2_inference_engine(logger)
            inference_engine.cleanup()
        except ImportError:
            if logger:
                logger.debug("Could not import inference engine for cleanup")
        
        # Cleanup model manager
        model_manager = get_seedvr2_model_manager(logger)
        model_manager.cleanup()
        
        if logger:
            logger.info("SeedVR2 resources cleaned up successfully")
            
    except Exception as e:
        if logger:
            logger.warning(f"Error during SeedVR2 cleanup: {e}")

# GPU detection utilities for multi-GPU support
def util_detect_available_gpus(logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Detect available CUDA GPUs for multi-GPU support
    
    Args:
        logger: Optional logger instance
        
    Returns:
        List of GPU information dictionaries
    """
    gpus = []
    
    try:
        import torch
        if not torch.cuda.is_available():
            return gpus
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_gb = gpu_props.total_memory / (1024**3)
            
            gpu_info = {
                'index': i,
                'name': gpu_props.name,
                'memory_gb': memory_gb,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}",
                'multiprocessor_count': gpu_props.multi_processor_count,
                'display_name': f"GPU {i}: {gpu_props.name} ({memory_gb:.1f}GB)"
            }
            
            gpus.append(gpu_info)
            
    except Exception as e:
        if logger:
            logger.warning(f"Error detecting GPUs: {e}")
    
    return gpus

def util_validate_gpu_selection(gpu_devices: str) -> Tuple[bool, List[str]]:
    """
    Validate GPU device selection string
    
    Args:
        gpu_devices: Comma-separated GPU device indices (e.g., "0,1,2")
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if not gpu_devices.strip():
        errors.append("GPU device selection cannot be empty")
        return False, errors
    
    try:
        # Parse device indices
        device_indices = [int(x.strip()) for x in gpu_devices.split(',') if x.strip()]
        
        if not device_indices:
            errors.append("No valid GPU indices found")
            return False, errors
        
        # Check for duplicates
        if len(device_indices) != len(set(device_indices)):
            errors.append("Duplicate GPU indices found")
        
        # Check range
        available_gpus = util_detect_available_gpus()
        max_gpu_index = len(available_gpus) - 1
        
        for idx in device_indices:
            if idx < 0 or idx > max_gpu_index:
                errors.append(f"GPU index {idx} is out of range (0-{max_gpu_index})")
        
    except ValueError:
        errors.append("Invalid GPU device format. Use comma-separated integers (e.g., '0,1,2')")
    
    is_valid = len(errors) == 0
    return is_valid, errors 
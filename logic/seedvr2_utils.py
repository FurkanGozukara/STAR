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

def util_scan_seedvr2_models(logger: Optional[logging.Logger] = None, include_missing: bool = False) -> List[Dict[str, Any]]:
    """
    Scan for available SeedVR2 models
    
    Args:
        logger: Optional logger instance
        include_missing: If True, include models that don't exist on disk (default: False)
        
    Returns:
        List of model dictionaries
    """
    if not SEEDVR2_AVAILABLE:
        return []
    
    model_manager = get_seedvr2_model_manager(logger)
    return model_manager.scan_available_models(include_missing=include_missing)

def util_get_model_block_count(model_filename: str) -> int:
    """
    Get the number of blocks in a SeedVR2 model based on its name.
    
    Args:
        model_filename: The model filename to check
        
    Returns:
        Number of blocks in the model
    """
    if not model_filename:
        return 20  # Default safe value
    
    model_lower = model_filename.lower()
    if '3b' in model_lower:
        return 32  # 3B models have 32 blocks
    elif '7b' in model_lower:
        return 36  # 7B models have 36 blocks (estimated, adjust if different)
    else:
        return 20  # Safe default for unknown models

def util_format_model_info_display(model_filename: str, logger: Optional[logging.Logger] = None) -> str:
    """
    Format model information for display in UI
    
    Args:
        model_filename: Model filename
        logger: Optional logger instance
        
    Returns:
        Formatted information string for UI display
    """
    if not model_filename:
        return "No model selected"
    
    try:
        # Get model information
        model_info = util_get_seedvr2_model_info(model_filename, logger)
        
        if 'error' in model_info:
            return f"❌ Error loading model info: {model_info['error']}"
        
        # Extract information
        display_name = model_info.get('display_name', model_filename)
        variant = model_info.get('variant', 'Unknown')
        precision = model_info.get('precision', 'Unknown')
        params = model_info.get('params', 'Unknown')
        recommended_vram = model_info.get('recommended_vram_gb', 8)
        description = model_info.get('description', '')
        available = model_info.get('available', False)
        size_mb = model_info.get('size_mb', 0)
        
        # Create formatted display
        display_text = f"""📋 **{display_name}**

🔧 **Specifications:**
• Parameters: {params}
• Precision: {precision}
• Variant: {variant}

💾 **Memory Requirements:**
• Recommended VRAM: {recommended_vram:.1f} GB
• Model Size: {size_mb:.0f} MB
"""
        
        if description:
            display_text += f"\n📝 **Description:**\n{description}\n"
        
        # Add availability status
        if available:
            display_text += "\n✅ **Status:** Model available and ready to use"
        else:
            display_text += "\n📥 **Status:** Model not found - will be downloaded when needed"
        
        # Add performance recommendations
        vram_info = util_get_vram_info(logger)
        if 'error' not in vram_info:
            available_vram = vram_info.get('free_gb', 0)
            recommendations = util_get_block_swap_recommendations(model_filename, logger)
            
            display_text += f"\n\n🎯 **Recommendations for your system ({available_vram:.1f}GB VRAM):**"
            
            if recommendations.get('blocks_to_swap', 0) == 0:
                display_text += "\n• ✅ No optimization needed - sufficient VRAM"
                display_text += f"\n• 🚀 Recommended batch size: {max(8, int(available_vram / 3))}"
            else:
                blocks = recommendations.get('blocks_to_swap', 0)
                display_text += f"\n• 🔧 Enable Block Swap: {blocks} blocks"
                if recommendations.get('offload_io_components', False):
                    display_text += "\n• 💾 Enable I/O component offloading"
                display_text += f"\n• 📊 Recommended batch size: 5 (minimum for temporal consistency)"
        
        # Add performance tips
        display_text += """\n\n💡 **Performance Tips:**
• Use batch size ≥ 5 for temporal consistency
• Enable Flash Attention for 15-20% speedup
• Use FP8 models for better VRAM efficiency
• Consider block swap for low VRAM systems"""
        
        return display_text
        
    except Exception as e:
        error_msg = f"Error formatting model info: {e}"
        if logger:
            logger.error(error_msg)
        return f"❌ {error_msg}"

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

def util_validate_seedvr2_model(model_filename: str, logger: Optional[logging.Logger] = None) -> Tuple[bool, str]:
    """
    Validate a SeedVR2 model file
    
    Args:
        model_filename: Name of the model file to validate
        logger: Optional logger instance
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if not model_filename:
        return False, "No model filename provided"
    
    try:
        # Check if SeedVR2 is available
        if not SEEDVR2_AVAILABLE:
            return False, "SeedVR2 modules not available"
        
        # Get model manager and check if model exists
        model_manager = get_seedvr2_model_manager(logger)
        model_info = model_manager.get_model_info(model_filename)
        
        if not model_info:
            return False, f"Model information not found for {model_filename}"
        
        # Check if model file exists
        model_path = model_info.get('path')
        if not model_path or not os.path.exists(model_path):
            return False, f"Model file not found: {model_filename}"
        
        # Check file size (models should be at least 100MB)
        try:
            file_size = os.path.getsize(model_path)
            if file_size < 100 * 1024 * 1024:  # 100MB minimum
                return False, f"Model file appears corrupted (too small): {file_size / (1024*1024):.1f} MB"
        except Exception as e:
            return False, f"Error checking model file size: {e}"
        
        # Check file extension
        if not model_filename.endswith('.safetensors'):
            return False, f"Invalid model format. Expected .safetensors, got: {model_filename}"
        
        # Additional model-specific validation
        variant = model_info.get('variant', 'unknown')
        precision = model_info.get('precision', 'unknown')
        
        if variant == 'unknown' or precision == 'unknown':
            if logger:
                logger.warning(f"Could not determine model variant/precision for {model_filename}")
        
        # Check VRAM requirements vs available VRAM
        vram_info = util_get_vram_info(logger)
        if 'error' not in vram_info:
            available_vram = vram_info.get('free_gb', 0)
            recommended_vram = model_info.get('recommended_vram_gb', 8)
            
            if available_vram < recommended_vram * 0.5:  # Less than half the recommended VRAM
                return False, f"Insufficient VRAM: {available_vram:.1f}GB available, {recommended_vram:.1f}GB recommended"
        
        if logger:
            logger.info(f"Model validation successful: {model_filename}")
        
        return True, f"Model validation successful: {variant} {precision}"
        
    except Exception as e:
        error_msg = f"Model validation error: {e}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

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
    from .star_dataclasses import DEFAULT_SEEDVR2_DEFAULT_RESOLUTION
    resolution = config_dict.get('seedvr2_resolution', DEFAULT_SEEDVR2_DEFAULT_RESOLUTION)
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

def util_get_recommended_settings_for_vram(total_vram_gb: float, model_filename: str, 
                                          target_quality: str = "balanced", 
                                          logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Get recommended SeedVR2 settings based on available VRAM and model
    
    Args:
        total_vram_gb: Total available VRAM in GB
        model_filename: Selected model filename
        target_quality: Target quality level ("fast", "balanced", "quality")
        logger: Optional logger instance
        
    Returns:
        Dictionary with recommended settings
    """
    try:
        # Parse model info from filename
        if '3b' in model_filename.lower():
            model_size = '3B'
            base_vram_requirement = 6.0  # GB
        elif '7b' in model_filename.lower():
            model_size = '7B'
            base_vram_requirement = 12.0  # GB
        else:
            model_size = 'Unknown'
            base_vram_requirement = 8.0  # GB (default estimate)
        
        # Adjust for precision
        if 'fp8' in model_filename.lower():
            precision = 'FP8'
            vram_multiplier = 0.7  # FP8 uses less VRAM
        else:
            precision = 'FP16'
            vram_multiplier = 1.0
        
        estimated_vram_needed = base_vram_requirement * vram_multiplier
        
        # Basic recommendations based on VRAM availability
        if total_vram_gb >= estimated_vram_needed * 1.5:
            # Plenty of VRAM
            recommendations = {
                'batch_size': 8 if target_quality == "quality" else 6,
                'temporal_overlap': 3,
                'enable_block_swap': False,
                'block_swap_counter': 0,
                'preserve_vram': False,
                'enable_multi_gpu': False,
                'flash_attention': True,
                'color_correction': True,
                'enable_frame_padding': True
            }
        elif total_vram_gb >= estimated_vram_needed:
            # Adequate VRAM
            recommendations = {
                'batch_size': 6 if target_quality == "quality" else 5,
                'temporal_overlap': 2,
                'enable_block_swap': False,
                'block_swap_counter': 0,
                'preserve_vram': True,
                'enable_multi_gpu': False,
                'flash_attention': True,
                'color_correction': True,
                'enable_frame_padding': True
            }
        else:
            # Limited VRAM - enable optimizations
            block_swap_blocks = min(16, max(4, int((estimated_vram_needed - total_vram_gb) * 3)))
            recommendations = {
                'batch_size': 5,  # Minimum for temporal consistency
                'temporal_overlap': 1,
                'enable_block_swap': True,
                'block_swap_counter': block_swap_blocks,
                'preserve_vram': True,
                'enable_multi_gpu': False,
                'flash_attention': True,
                'color_correction': True,
                'enable_frame_padding': False  # Disable for VRAM savings
            }
        
        # Quality-based adjustments
        if target_quality == "fast":
            recommendations['batch_size'] = max(5, recommendations['batch_size'] - 1)
            recommendations['temporal_overlap'] = max(1, recommendations['temporal_overlap'] - 1)
        elif target_quality == "quality":
            recommendations['batch_size'] = min(12, recommendations['batch_size'] + 2)
            recommendations['temporal_overlap'] = min(4, recommendations['temporal_overlap'] + 1)
        
        # Add metadata
        recommendations.update({
            'model_size': model_size,
            'precision': precision,
            'estimated_vram_gb': estimated_vram_needed,
            'available_vram_gb': total_vram_gb,
            'vram_ratio': total_vram_gb / estimated_vram_needed if estimated_vram_needed > 0 else 0,
            'target_quality': target_quality
        })
        
        if logger:
            logger.info(f"Generated SeedVR2 recommendations for {model_filename}: batch_size={recommendations['batch_size']}, block_swap={recommendations['enable_block_swap']}")
        
        return recommendations
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to generate recommendations: {e}")
        return {
            'batch_size': 5,
            'temporal_overlap': 2,
            'enable_block_swap': False,
            'block_swap_counter': 0,
            'preserve_vram': True,
            'enable_multi_gpu': False,
            'flash_attention': True,
            'color_correction': True,
            'enable_frame_padding': True,
            'model_size': 'Unknown',
            'precision': 'Unknown',
            'estimated_vram_gb': 0,
            'available_vram_gb': total_vram_gb,
            'target_quality': target_quality
        }

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
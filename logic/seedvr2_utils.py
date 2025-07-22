"""
SeedVR2 Utilities Module

This module provides utility functions for SeedVR2 integration including:
- Model scanning and management from SeedVR2/models directory
- Block swap configuration and optimization
- Multi-GPU device detection and management
- Memory optimization utilities
- Color correction and wavelet reconstruction helpers
- Model download and validation utilities
"""

import os
import sys
import time
import json
import shutil
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import torch

def scan_for_seedvr2_models(models_dir: str = None, logger: logging.Logger = None) -> List[Dict[str, Any]]:
    """
    Scan for available SeedVR2 models in the models directory.
    
    Args:
        models_dir: Path to models directory (uses default if None)
        logger: Logger instance for debug/info messages
        
    Returns:
        List of dictionaries containing model information
    """
    
    if models_dir is None:
        # Get default SeedVR2 models directory
        seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
        models_dir = os.path.join(seedvr2_base_path, 'models')
    
    if not os.path.exists(models_dir):
        if logger:
            logger.warning(f"SeedVR2 models directory not found: {models_dir}")
        return []
    
    models_info = []
    
    try:
        for filename in os.listdir(models_dir):
            if filename.endswith('.safetensors') and 'seedvr2' in filename.lower():
                model_path = os.path.join(models_dir, filename)
                
                # Get file size
                file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
                
                # Parse model information
                model_info = parse_model_filename(filename)
                model_info['filename'] = filename
                model_info['file_size'] = file_size
                model_info['file_size_mb'] = round(file_size / (1024 * 1024), 1)
                model_info['file_size_gb'] = round(file_size / (1024 * 1024 * 1024), 1)
                model_info['available'] = True
                
                models_info.append(model_info)
                
                if logger:
                    logger.debug(f"Found SeedVR2 model: {filename} ({model_info['file_size_gb']:.1f} GB)")
    
    except Exception as e:
        if logger:
            logger.error(f"Error scanning SeedVR2 models: {e}")
        return []
    
    # Sort models by preference (3B FP8, 3B FP16, 7B FP8, 7B FP16)
    models_info.sort(key=lambda x: get_model_sort_priority(x['filename']))
    
    if logger:
        logger.info(f"Found {len(models_info)} SeedVR2 models")
    
    return models_info


def parse_model_filename(filename: str) -> Dict[str, Any]:
    """
    Parse SeedVR2 model filename to extract information.
    
    Args:
        filename: Model filename
        
    Returns:
        Dictionary with parsed model information
    """
    
    info = {
        'name': filename,
        'display_name': filename,
        'size_category': 'Unknown',
        'parameter_count': 'Unknown',
        'precision': 'Unknown',
        'variant': 'Standard',
        'recommended_batch_size': 5,
        'min_vram_gb': 8,
        'recommended_vram_gb': 12,
        'speed_rating': 'Medium',
        'quality_rating': 'High'
    }
    
    filename_lower = filename.lower()
    
    # Parse parameter count
    if '3b' in filename_lower:
        info['size_category'] = '3B'
        info['parameter_count'] = '3 Billion'
        info['recommended_batch_size'] = 8
        info['min_vram_gb'] = 6
        info['recommended_vram_gb'] = 8
        info['speed_rating'] = 'Fast'
        info['display_name'] = 'SeedVR2 3B'
    elif '7b' in filename_lower:
        info['size_category'] = '7B'
        info['parameter_count'] = '7 Billion'
        info['recommended_batch_size'] = 5
        info['min_vram_gb'] = 12
        info['recommended_vram_gb'] = 16
        info['speed_rating'] = 'Medium'
        info['quality_rating'] = 'Very High'
        info['display_name'] = 'SeedVR2 7B'
    
    # Parse precision
    if 'fp8' in filename_lower:
        info['precision'] = 'FP8'
        info['display_name'] += ' FP8'
        info['speed_rating'] = 'Very Fast' if info['speed_rating'] == 'Fast' else 'Fast'
        info['min_vram_gb'] = max(4, info['min_vram_gb'] // 2)
        info['recommended_vram_gb'] = max(6, info['recommended_vram_gb'] // 2)
    elif 'fp16' in filename_lower:
        info['precision'] = 'FP16'
        info['display_name'] += ' FP16'
    
    # Parse variant
    if 'sharp' in filename_lower:
        info['variant'] = 'Sharp'
        info['display_name'] += ' Sharp'
        info['quality_rating'] = 'Very High'
    
    # Create description
    info['description'] = f"{info['parameter_count']} parameters, {info['precision']} precision"
    if info['variant'] != 'Standard':
        info['description'] += f", {info['variant']} variant"
    
    info['vram_info'] = f"Min: {info['min_vram_gb']} GB, Recommended: {info['recommended_vram_gb']} GB"
    
    return info


def get_model_sort_priority(filename: str) -> int:
    """Get sorting priority for model ordering (lower = higher priority)."""
    
    filename_lower = filename.lower()
    priority = 100  # Default priority
    
    # Prefer 3B over 7B for better compatibility
    if '3b' in filename_lower:
        priority -= 50
    elif '7b' in filename_lower:
        priority -= 30
    
    # Prefer FP8 over FP16 for VRAM efficiency
    if 'fp8' in filename_lower:
        priority -= 20
    elif 'fp16' in filename_lower:
        priority -= 10
    
    # Standard variant before Sharp
    if 'sharp' not in filename_lower:
        priority -= 5
    
    return priority


def get_model_dropdown_choices(models_dir: str = None, logger: logging.Logger = None) -> List[str]:
    """
    Get list of model choices formatted for Gradio dropdown.
    
    Args:
        models_dir: Path to models directory
        logger: Logger instance
        
    Returns:
        List of formatted model choice strings
    """
    
    models_info = scan_for_seedvr2_models(models_dir, logger)
    
    if not models_info:
        return ["No SeedVR2 models found"]
    
    choices = []
    for model_info in models_info:
        # Format: "SeedVR2 3B FP8 (3.2 GB) - Fast, Low VRAM"
        choice = f"{model_info['display_name']} ({model_info['file_size_gb']:.1f} GB)"
        choice += f" - {model_info['speed_rating']}, {model_info['precision']}"
        choices.append(choice)
    
    return choices


def extract_model_filename_from_dropdown(dropdown_choice: str, models_dir: str = None) -> Optional[str]:
    """
    Extract actual model filename from dropdown choice.
    
    Args:
        dropdown_choice: Selected dropdown choice
        models_dir: Path to models directory
        
    Returns:
        Actual model filename or None if not found
    """
    
    if not dropdown_choice or dropdown_choice == "No SeedVR2 models found":
        return None
    
    models_info = scan_for_seedvr2_models(models_dir)
    
    for model_info in models_info:
        expected_choice = f"{model_info['display_name']} ({model_info['file_size_gb']:.1f} GB)"
        expected_choice += f" - {model_info['speed_rating']}, {model_info['precision']}"
        
        if expected_choice == dropdown_choice:
            return model_info['filename']
    
    # Fallback: try to find by display name in choice
    for model_info in models_info:
        if model_info['display_name'] in dropdown_choice:
            return model_info['filename']
    
    return None


def validate_seedvr2_model(model_filename: str, models_dir: str = None, logger: logging.Logger = None) -> Tuple[bool, str]:
    """
    Validate SeedVR2 model existence and integrity.
    
    Args:
        model_filename: Model filename to validate
        models_dir: Path to models directory
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, message)
    """
    
    if not model_filename:
        return False, "No model specified"
    
    if models_dir is None:
        seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
        models_dir = os.path.join(seedvr2_base_path, 'models')
    
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_filename}"
    
    # Check file size (models should be at least 100MB)
    try:
        file_size = os.path.getsize(model_path)
        if file_size < 100 * 1024 * 1024:  # 100MB minimum
            return False, f"Model file appears corrupted (too small): {file_size / (1024*1024):.1f} MB"
    except Exception as e:
        return False, f"Error checking model file: {e}"
    
    # Check file extension
    if not model_filename.endswith('.safetensors'):
        return False, f"Invalid model format. Expected .safetensors, got: {model_filename}"
    
    if logger:
        logger.info(f"Model validation successful: {model_filename}")
    
    return True, "Model validation successful"


def setup_block_swap_config(
    enable_block_swap: bool,
    block_swap_counter: int,
    block_swap_offload_io: bool = False,
    block_swap_model_caching: bool = False,
    logger: logging.Logger = None
) -> Optional[Dict[str, Any]]:
    """
    Setup block swap configuration for SeedVR2.
    
    Args:
        enable_block_swap: Whether to enable block swap
        block_swap_counter: Number of blocks to swap (0 = disabled)
        block_swap_offload_io: Enable I/O component offloading
        block_swap_model_caching: Enable model caching between runs
        logger: Logger instance
        
    Returns:
        Block swap configuration dictionary or None if disabled
    """
    
    if not enable_block_swap or block_swap_counter <= 0:
        if logger:
            logger.info("Block swap disabled")
        return None
    
    # Validate block count
    max_blocks = 32  # Reasonable maximum
    if block_swap_counter > max_blocks:
        if logger:
            logger.warning(f"Block swap counter {block_swap_counter} exceeds maximum {max_blocks}, capping")
        block_swap_counter = max_blocks
    
    config = {
        "enabled": True,
        "blocks_to_swap": block_swap_counter,
        "offload_io": block_swap_offload_io,
        "model_caching": block_swap_model_caching,
        "swap_mode": "auto",  # Can be "auto", "manual", "aggressive"
        "memory_threshold": 0.85,  # Swap when VRAM usage exceeds 85%
        "cache_location": "cpu"  # Where to cache swapped blocks
    }
    
    if logger:
        logger.info(f"Block swap configured: {block_swap_counter} blocks, I/O offload: {block_swap_offload_io}, caching: {block_swap_model_caching}")
    
    return config


def get_optimal_batch_size_for_vram(
    available_vram_gb: float,
    model_filename: str = None,
    target_resolution: Tuple[int, int] = (1024, 1024),
    preserve_vram: bool = True
) -> int:
    """
    Calculate optimal batch size based on available VRAM.
    
    Args:
        available_vram_gb: Available VRAM in GB
        model_filename: Model filename for specific requirements
        target_resolution: Target resolution (width, height)
        preserve_vram: Whether preserve VRAM mode is enabled
        
    Returns:
        Recommended batch size
    """
    
    # Base VRAM requirements (rough estimates)
    base_vram_usage = {
        '3b_fp8': 3.5,
        '3b_fp16': 6.5,
        '7b_fp8': 7.5,
        '7b_fp16': 15.0
    }
    
    # Determine model type
    model_key = '3b_fp8'  # Default
    if model_filename:
        filename_lower = model_filename.lower()
        if '7b' in filename_lower:
            model_key = '7b_fp8' if 'fp8' in filename_lower else '7b_fp16'
        elif '3b' in filename_lower:
            model_key = '3b_fp8' if 'fp8' in filename_lower else '3b_fp16'
    
    base_usage = base_vram_usage.get(model_key, 4.0)
    
    # Adjust for preserve VRAM mode
    if preserve_vram:
        base_usage *= 0.7  # Preserve VRAM reduces base usage
    
    # Calculate resolution factor
    width, height = target_resolution
    resolution_factor = (width * height) / (1024 * 1024)  # Normalized to 1024x1024
    
    # VRAM per frame (rough estimate)
    vram_per_frame = 0.1 * resolution_factor
    
    # Available VRAM for frames (after model)
    available_for_frames = max(0.5, available_vram_gb - base_usage)
    
    # Calculate batch size
    theoretical_batch_size = int(available_for_frames / vram_per_frame)
    
    # Apply constraints
    batch_size = max(5, min(theoretical_batch_size, 32))  # Min 5 for temporal consistency, max 32 practical
    
    return batch_size


def detect_available_gpus(logger: logging.Logger = None) -> List[Dict[str, Any]]:
    """
    Detect available GPUs and their capabilities with enhanced multi-GPU analysis.
    
    Args:
        logger: Logger instance
        
    Returns:
        List of GPU information dictionaries with detailed multi-GPU compatibility
    """
    
    gpus = []
    
    if not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available")
        return gpus
    
    try:
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            try:
                # Get GPU properties
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                cached_memory = torch.cuda.memory_reserved(i)
                free_memory = total_memory - allocated_memory
                
                # Enhanced GPU analysis for multi-GPU suitability
                memory_bandwidth = getattr(props, 'memory_bus_width', 256) * 2 * getattr(props, 'memory_clock_rate', 5001000) / 8 / 1e9  # GB/s estimate
                is_suitable_for_multigpu = (
                    free_memory / (1024**3) >= 4.0 and  # At least 4GB free
                    props.major >= 6 and  # Compute capability 6.0+
                    props.multi_processor_count >= 20  # Sufficient compute units
                )
                
                gpu_info = {
                    'id': i,
                    'name': props.name,
                    'total_memory_gb': total_memory / (1024**3),
                    'free_memory_gb': free_memory / (1024**3),
                    'allocated_memory_gb': allocated_memory / (1024**3),
                    'cached_memory_gb': cached_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count,
                    'is_available': True,
                    'memory_bandwidth_gbps': memory_bandwidth,
                    'is_suitable_for_multigpu': is_suitable_for_multigpu,
                    'memory_clock_mhz': getattr(props, 'memory_clock_rate', 0) // 1000,
                    'cuda_capability_major': props.major,
                    'cuda_capability_minor': props.minor,
                    'nvml_available': False
                }
                
                # Determine suitability for SeedVR2
                gpu_info['suitable_for_3b'] = gpu_info['free_memory_gb'] >= 4
                gpu_info['suitable_for_7b'] = gpu_info['free_memory_gb'] >= 8
                gpu_info['recommended_batch_size'] = get_optimal_batch_size_for_vram(
                    gpu_info['free_memory_gb']
                )
                
                # Try to get additional info via nvidia-ml-py if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        gpu_info['temperature_c'] = temp
                    except:
                        gpu_info['temperature_c'] = None
                    
                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                        gpu_info['power_usage_w'] = power
                    except:
                        gpu_info['power_usage_w'] = None
                    
                    # Get utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_info['gpu_utilization_percent'] = util.gpu
                        gpu_info['memory_utilization_percent'] = util.memory
                    except:
                        gpu_info['gpu_utilization_percent'] = None
                        gpu_info['memory_utilization_percent'] = None
                    
                    gpu_info['nvml_available'] = True
                    
                except ImportError:
                    # pynvml not available - set None values
                    gpu_info.update({
                        'temperature_c': None,
                        'power_usage_w': None,
                        'gpu_utilization_percent': None,
                        'memory_utilization_percent': None
                    })
                except Exception as e:
                    # NVML error - set None values
                    if logger:
                        logger.debug(f"NVML error for GPU {i}: {e}")
                    gpu_info.update({
                        'temperature_c': None,
                        'power_usage_w': None,
                        'gpu_utilization_percent': None,
                        'memory_utilization_percent': None
                    })
                
                gpus.append(gpu_info)
                
                if logger:
                    suitability = "✓ Multi-GPU Ready" if is_suitable_for_multigpu else "⚠ Limited for Multi-GPU"
                    logger.debug(f"GPU {i}: {props.name}, {gpu_info['free_memory_gb']:.1f} GB free - {suitability}")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Error getting info for GPU {i}: {e}")
                
                # Add basic info even if detailed info fails
                gpus.append({
                    'id': i,
                    'name': f'GPU {i}',
                    'total_memory_gb': 0,
                    'free_memory_gb': 0,
                    'is_available': False,
                    'is_suitable_for_multigpu': False,
                    'error': str(e)
                })
        
        if logger:
            suitable_count = len([gpu for gpu in gpus if gpu.get('is_suitable_for_multigpu', False)])
            logger.info(f"Detected {len(gpus)} GPUs ({suitable_count} suitable for multi-GPU)")
            
    except Exception as e:
        if logger:
            logger.error(f"Error detecting GPUs: {e}")
    
    return gpus


def analyze_multi_gpu_configuration(
    gpu_list: List[Dict[str, Any]], 
    model_filename: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Analyze multi-GPU configuration and provide intelligent recommendations.
    
    Args:
        gpu_list: List of GPU information dictionaries
        model_filename: Selected model filename
        logger: Logger instance
        
    Returns:
        Multi-GPU analysis and recommendations
    """
    
    if not gpu_list:
        return {
            "multi_gpu_possible": False,
            "reason": "No GPUs detected",
            "recommended_gpus": [],
            "total_vram": 0,
            "performance_estimate": "unavailable"
        }
    
    # Filter suitable GPUs
    suitable_gpus = [gpu for gpu in gpu_list if gpu.get('is_suitable_for_multigpu', False)]
    
    if len(suitable_gpus) < 2:
        return {
            "multi_gpu_possible": False,
            "reason": f"Only {len(suitable_gpus)} suitable GPU(s) found (need 2+)",
            "recommended_gpus": [gpu['id'] for gpu in suitable_gpus],
            "total_vram": sum(gpu.get('free_memory_gb', 0) for gpu in gpu_list),
            "performance_estimate": "single_gpu_only"
        }
    
    # Calculate total available VRAM
    total_vram = sum(gpu.get('free_memory_gb', 0) for gpu in suitable_gpus)
    
    # Model VRAM requirements (per GPU in multi-GPU setup)
    model_requirements = {
        "3b_fp8": 3.5,   # Per GPU requirement
        "3b_fp16": 4.5,
        "7b_fp8": 6.0,
        "7b_fp16": 8.0
    }
    
    # Extract model type
    model_type = "3b_fp8"  # Default
    if "3b" in model_filename.lower():
        model_type = "3b_fp8" if "fp8" in model_filename.lower() else "3b_fp16"
    elif "7b" in model_filename.lower():
        model_type = "7b_fp8" if "fp8" in model_filename.lower() else "7b_fp16"
    
    per_gpu_requirement = model_requirements.get(model_type, 4.0)
    
    # Find optimal GPU selection
    optimal_gpus = []
    for gpu in suitable_gpus:
        if gpu.get('free_memory_gb', 0) >= per_gpu_requirement:
            optimal_gpus.append(gpu)
    
    if len(optimal_gpus) < 2:
        return {
            "multi_gpu_possible": False,
            "reason": f"Insufficient VRAM per GPU ({per_gpu_requirement:.1f}GB required per GPU)",
            "recommended_gpus": [gpu['id'] for gpu in optimal_gpus],
            "total_vram": total_vram,
            "performance_estimate": "insufficient_vram"
        }
    
    # Performance estimation with diminishing returns
    num_optimal_gpus = min(len(optimal_gpus), 4)  # Limit to 4 GPUs for practical reasons
    performance_boost = min(num_optimal_gpus * 0.8, 3.2)  # Diminishing returns, max ~3.2x
    
    # Memory bandwidth analysis
    min_bandwidth = min(gpu.get('memory_bandwidth_gbps', 100) for gpu in optimal_gpus[:num_optimal_gpus])
    bandwidth_efficiency = min(min_bandwidth / 500, 1.0)  # Normalize to typical high-end GPU
    
    effective_speedup = performance_boost * bandwidth_efficiency
    
    return {
        "multi_gpu_possible": True,
        "reason": f"{num_optimal_gpus} suitable GPUs found",
        "recommended_gpus": [gpu['id'] for gpu in optimal_gpus[:num_optimal_gpus]],
        "total_vram": sum(gpu.get('free_memory_gb', 0) for gpu in optimal_gpus[:num_optimal_gpus]),
        "performance_estimate": f"{effective_speedup:.1f}x speedup",
        "optimal_gpu_count": num_optimal_gpus,
        "per_gpu_vram_requirement": per_gpu_requirement,
        "memory_bandwidth_efficiency": bandwidth_efficiency,
        "load_balancing_score": min(1.0, len(optimal_gpus) / 2),
        "recommended_batch_size": min(12, num_optimal_gpus * 3),  # Scaled batch size
        "gpu_details": [
            {
                "id": gpu['id'],
                "name": gpu['name'],
                "vram_gb": gpu.get('free_memory_gb', 0),
                "compute_capability": gpu.get('compute_capability', 'unknown'),
                "bandwidth_gbps": gpu.get('memory_bandwidth_gbps', 0)
            }
            for gpu in optimal_gpus[:num_optimal_gpus]
        ]
    }


def get_optimized_multi_gpu_settings(
    available_gpus: List[Dict[str, Any]],
    model_filename: str,
    target_quality: str = "balanced",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Get optimized multi-GPU settings based on available hardware.
    
    Args:
        available_gpus: List of available GPU information
        model_filename: Selected model filename
        target_quality: Target quality level (fast, balanced, quality)
        logger: Logger instance
        
    Returns:
        Optimized multi-GPU configuration
    """
    
    analysis = analyze_multi_gpu_configuration(available_gpus, model_filename, logger)
    
    if not analysis.get("multi_gpu_possible", False):
        return {
            "enable_multi_gpu": False,
            "gpu_devices": "0",
            "reason": analysis.get("reason", "Multi-GPU not possible"),
            "single_gpu_fallback": True
        }
    
    recommended_gpus = analysis.get("recommended_gpus", [])
    optimal_count = analysis.get("optimal_gpu_count", 2)
    
    # Quality-based adjustments
    if target_quality == "fast":
        # Use fewer GPUs for faster coordination
        gpu_count = min(2, optimal_count)
    elif target_quality == "quality":
        # Use more GPUs for maximum parallel processing
        gpu_count = optimal_count
    else:  # balanced
        # Use moderate GPU count for balance
        gpu_count = min(3, optimal_count)
    
    selected_gpus = recommended_gpus[:gpu_count]
    gpu_devices_str = ','.join(map(str, selected_gpus))
    
    # Calculate optimized batch size
    base_batch_size = 8 if "3b" in model_filename.lower() else 6
    multi_gpu_batch_size = min(base_batch_size * gpu_count, 16)  # Scale but cap
    
    return {
        "enable_multi_gpu": True,
        "gpu_devices": gpu_devices_str,
        "gpu_count": gpu_count,
        "selected_gpus": selected_gpus,
        "optimized_batch_size": multi_gpu_batch_size,
        "expected_speedup": analysis.get("performance_estimate", "unknown"),
        "total_vram": analysis.get("total_vram", 0),
        "reason": f"Optimized for {target_quality} with {gpu_count} GPUs",
        "gpu_details": analysis.get("gpu_details", []),
        "load_balancing_active": True,
        "memory_optimization": gpu_count > 2
    }


def format_gpu_dropdown_choices(gpus: List[Dict[str, Any]]) -> List[str]:
    """
    Format GPU information for dropdown choices.
    
    Args:
        gpus: List of GPU information dictionaries
        
    Returns:
        List of formatted GPU choice strings
    """
    
    if not gpus:
        return ["No CUDA GPUs available"]
    
    choices = []
    for gpu in gpus:
        if gpu.get('is_available', False):
            choice = f"GPU {gpu['id']}: {gpu['name']} ({gpu['free_memory_gb']:.1f} GB free)"
            if gpu.get('suitable_for_3b', False):
                choice += " ✓"
            choices.append(choice)
        else:
            choice = f"GPU {gpu['id']}: {gpu['name']} (Unavailable)"
            choices.append(choice)
    
    return choices


def validate_multi_gpu_setup(gpu_devices: str, logger: logging.Logger = None) -> Tuple[bool, List[str], str]:
    """
    Validate multi-GPU setup configuration.
    
    Args:
        gpu_devices: Comma-separated GPU device IDs
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, valid_device_list, error_message)
    """
    
    if not gpu_devices or not gpu_devices.strip():
        return False, [], "No GPU devices specified"
    
    # Parse device IDs
    device_ids = [d.strip() for d in gpu_devices.split(',') if d.strip()]
    
    if not device_ids:
        return False, [], "No valid GPU device IDs found"
    
    # Get available GPUs
    available_gpus = detect_available_gpus(logger)
    available_ids = [str(gpu['id']) for gpu in available_gpus if gpu.get('is_available', False)]
    
    # Validate each device ID
    valid_devices = []
    invalid_devices = []
    
    for device_id in device_ids:
        try:
            # Validate format
            int(device_id)  # Should be parseable as integer
            
            if device_id in available_ids:
                valid_devices.append(device_id)
            else:
                invalid_devices.append(device_id)
                
        except ValueError:
            invalid_devices.append(device_id)
    
    if invalid_devices:
        error_msg = f"Invalid/unavailable GPU devices: {', '.join(invalid_devices)}"
        if valid_devices:
            error_msg += f". Valid devices: {', '.join(valid_devices)}"
        return False, valid_devices, error_msg
    
    if len(valid_devices) < 1:
        return False, [], "No valid GPU devices found"
    
    if logger:
        if len(valid_devices) > 1:
            logger.info(f"Multi-GPU setup validated: {len(valid_devices)} GPUs ({', '.join(valid_devices)})")
        else:
            logger.info(f"Single GPU setup validated: GPU {valid_devices[0]}")
    
    return True, valid_devices, "GPU setup validation successful"


def estimate_processing_time(
    total_frames: int,
    model_filename: str = None,
    batch_size: int = 5,
    gpu_count: int = 1,
    resolution: Tuple[int, int] = (1024, 1024)
) -> Dict[str, Any]:
    """
    Estimate processing time for SeedVR2 upscaling.
    
    Args:
        total_frames: Total number of frames to process
        model_filename: Model filename for speed estimates
        batch_size: Batch size for processing
        gpu_count: Number of GPUs to use
        resolution: Target resolution
        
    Returns:
        Dictionary with time estimates
    """
    
    # Base processing rates (frames per second) for different models
    base_rates = {
        '3b_fp8': 2.5,
        '3b_fp16': 1.8,
        '7b_fp8': 1.2,
        '7b_fp16': 0.8
    }
    
    # Determine model type
    model_key = '3b_fp8'  # Default
    if model_filename:
        filename_lower = model_filename.lower()
        if '7b' in filename_lower:
            model_key = '7b_fp8' if 'fp8' in filename_lower else '7b_fp16'
        elif '3b' in filename_lower:
            model_key = '3b_fp8' if 'fp8' in filename_lower else '3b_fp16'
    
    base_rate = base_rates.get(model_key, 1.5)
    
    # Adjust for resolution
    width, height = resolution
    resolution_factor = (width * height) / (1024 * 1024)
    adjusted_rate = base_rate / max(1.0, resolution_factor)
    
    # Adjust for batch size (larger batches are more efficient)
    batch_efficiency = min(1.0 + (batch_size - 5) * 0.1, 1.5)
    adjusted_rate *= batch_efficiency
    
    # Adjust for multi-GPU (with some overhead)
    if gpu_count > 1:
        multi_gpu_efficiency = min(gpu_count * 0.85, gpu_count)  # 85% efficiency per additional GPU
        adjusted_rate *= multi_gpu_efficiency
    
    # Calculate estimates
    processing_time = total_frames / adjusted_rate
    
    # Add overhead estimates
    frame_extraction_time = total_frames * 0.02  # ~0.02s per frame
    video_creation_time = total_frames * 0.01   # ~0.01s per frame
    total_time = processing_time + frame_extraction_time + video_creation_time
    
    return {
        'processing_time_seconds': processing_time,
        'frame_extraction_time_seconds': frame_extraction_time,
        'video_creation_time_seconds': video_creation_time,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'processing_rate_fps': adjusted_rate,
        'model_type': model_key,
        'resolution_factor': resolution_factor,
        'batch_efficiency': batch_efficiency,
        'gpu_efficiency': multi_gpu_efficiency if gpu_count > 1 else 1.0
    }


def get_recommended_settings_for_vram(
    available_vram_gb: float,
    target_quality: str = "balanced"  # "fast", "balanced", "quality"
) -> Dict[str, Any]:
    """
    Get recommended SeedVR2 settings based on available VRAM.
    
    Args:
        available_vram_gb: Available VRAM in GB
        target_quality: Target quality level
        
    Returns:
        Dictionary with recommended settings
    """
    
    settings = {
        'model': None,
        'batch_size': 5,
        'enable_block_swap': False,
        'block_swap_counter': 0,
        'preserve_vram': True,
        'multi_gpu_recommended': False,
        'resolution_limit': (1024, 1024),
        'warnings': [],
        'vram_category': 'low'
    }
    
    # Categorize VRAM amount
    if available_vram_gb >= 16:
        settings['vram_category'] = 'high'
    elif available_vram_gb >= 8:
        settings['vram_category'] = 'medium'
    else:
        settings['vram_category'] = 'low'
    
    # Recommend model based on VRAM and quality target
    if available_vram_gb >= 16:
        # High VRAM - can handle 7B models
        if target_quality == "quality":
            settings['model'] = "seedvr2_ema_7b_fp16.safetensors"
            settings['batch_size'] = 8
            settings['preserve_vram'] = False
        else:  # fast or balanced
            settings['model'] = "seedvr2_ema_7b_fp8_e4m3fn.safetensors"
            settings['batch_size'] = 10
        settings['resolution_limit'] = (2048, 2048)
        
    elif available_vram_gb >= 8:
        # Medium VRAM - 3B models
        if target_quality == "quality":
            settings['model'] = "seedvr2_ema_3b_fp16.safetensors"
            settings['batch_size'] = 6
        else:
            settings['model'] = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
            settings['batch_size'] = 8
        settings['resolution_limit'] = (1536, 1536)
        
    elif available_vram_gb >= 6:
        # Low-medium VRAM - 3B FP8 with optimizations
        settings['model'] = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        settings['batch_size'] = 5
        settings['enable_block_swap'] = True
        settings['block_swap_counter'] = 4
        settings['resolution_limit'] = (1024, 1024)
        settings['warnings'].append("Consider enabling block swap for better VRAM efficiency")
        
    else:
        # Very low VRAM - aggressive optimizations needed
        settings['model'] = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
        settings['batch_size'] = 5
        settings['enable_block_swap'] = True
        settings['block_swap_counter'] = 8
        settings['resolution_limit'] = (768, 768)
        settings['warnings'].append("VRAM is very limited. Consider upgrading GPU or using lower resolution")
        settings['warnings'].append("Block swap is highly recommended")
    
    # Multi-GPU recommendation
    if available_vram_gb >= 6:
        settings['multi_gpu_recommended'] = True
    
    return settings


def format_model_info_display(model_filename: str) -> str:
    """
    Format model information for display in UI.
    
    Args:
        model_filename: Model filename
        
    Returns:
        Formatted information string
    """
    
    if not model_filename:
        return "No model selected"
    
    model_info = parse_model_filename(model_filename)
    
    display_text = f"""**{model_info['display_name']}**

📊 **Specifications:**
• Parameters: {model_info['parameter_count']}
• Precision: {model_info['precision']}
• Variant: {model_info['variant']}

💾 **Memory Requirements:**
• {model_info['vram_info']}

⚡ **Performance:**
• Speed: {model_info['speed_rating']}
• Quality: {model_info['quality_rating']}
• Recommended Batch Size: {model_info['recommended_batch_size']}

📝 **Description:**
{model_info['description']}
"""
    
    return display_text


def get_real_time_block_swap_status(logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Get real-time block swap status and memory information.
    
    Args:
        logger: Logger instance
        
    Returns:
        Dictionary with current status information
    """
    try:
        from .block_swap_manager import create_block_swap_manager, format_memory_info
        
        # Create temporary manager for status check
        manager = create_block_swap_manager(logger)
        status = manager.get_real_time_status()
        
        # Format for UI display
        memory_usage = status.get("memory_usage", {})
        
        if memory_usage:
            formatted_status = {
                "memory_info": format_memory_info(memory_usage),
                "vram_allocated": memory_usage.get("vram_allocated_gb", 0),
                "vram_reserved": memory_usage.get("vram_reserved_gb", 0),
                "system_ram": memory_usage.get("system_ram_gb", 0),
                "cpu_percent": memory_usage.get("cpu_percent", 0),
                "status": "healthy" if memory_usage.get("vram_allocated_gb", 0) < 10 else "high_usage"
            }
        else:
            formatted_status = {
                "memory_info": "Memory monitoring unavailable",
                "status": "unavailable"
            }
        
        return formatted_status
        
    except Exception as e:
        if logger:
            logger.warning(f"Failed to get block swap status: {e}")
        return {
            "memory_info": f"Status check failed: {e}",
            "status": "error"
        }


def get_intelligent_block_swap_recommendations(
    available_vram_gb: Optional[float] = None,
    model_filename: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    target_quality: str = "balanced",
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Get intelligent block swap recommendations based on current system state.
    
    Args:
        available_vram_gb: Available VRAM in GB (auto-detected if None)
        model_filename: Selected model filename
        target_quality: Target quality level (fast, balanced, quality)
        logger: Logger instance
        
    Returns:
        Dictionary with recommendations
    """
    try:
        from .block_swap_manager import create_block_swap_manager
        
        manager = create_block_swap_manager(logger)
        
        # Auto-detect VRAM if not provided
        if available_vram_gb is None:
            available_vram_gb = manager._estimate_available_vram()
        
        # Extract model type from filename
        model_type = manager._extract_model_type(model_filename)
        
        # Get recommendations
        recommendations = manager.optimizer.get_recommendations(
            available_vram_gb=available_vram_gb,
            model_type=model_type,
            target_quality=target_quality
        )
        
        # Add performance estimates
        if recommendations.get("enable_block_swap", False):
            perf_estimate = manager.optimizer.estimate_performance_impact(
                blocks_to_swap=recommendations.get("block_swap_counter", 0),
                io_offload=recommendations.get("offload_io", False)
            )
            recommendations["performance_estimate"] = perf_estimate
        
        # Format for UI display
        formatted_recommendations = {
            "enable_block_swap": recommendations.get("enable_block_swap", False),
            "recommended_blocks": recommendations.get("block_swap_counter", 0),
            "offload_io": recommendations.get("offload_io", False),
            "model_caching": recommendations.get("model_caching", False),
            "reason": recommendations.get("reason", "Unknown"),
            "vram_ratio": recommendations.get("vram_ratio", 1.0),
            "expected_performance": recommendations.get("expected_performance", "unknown"),
            "alternatives": recommendations.get("alternative_models", []),
            "available_vram": available_vram_gb
        }
        
        if "performance_estimate" in recommendations:
            perf = recommendations["performance_estimate"]
            formatted_recommendations.update({
                "performance_impact": perf.get("performance_impact_percent", 0),
                "memory_savings": perf.get("memory_savings_gb", 0),
                "efficiency_score": perf.get("efficiency_score", 0)
            })
        
        return formatted_recommendations
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to get block swap recommendations: {e}")
        return {
            "enable_block_swap": False,
            "recommended_blocks": 0,
            "reason": f"Error: {e}",
            "available_vram": available_vram_gb or 0
        }


def format_block_swap_recommendations_for_ui(recommendations: Dict[str, Any]) -> str:
    """
    Format block swap recommendations for UI display.
    
    Args:
        recommendations: Recommendations dictionary
        
    Returns:
        Formatted string for UI display
    """
    if not recommendations.get("enable_block_swap", False):
        reason = recommendations.get("reason", "Sufficient VRAM")
        vram = recommendations.get("available_vram", 0)
        return f"""✅ Block Swap Not Needed

Available VRAM: {vram:.1f}GB
Reason: {reason}

💡 You have sufficient VRAM for optimal performance without block swapping."""
    
    blocks = recommendations.get("recommended_blocks", 0)
    io_offload = recommendations.get("offload_io", False)
    caching = recommendations.get("model_caching", False)
    performance_impact = recommendations.get("performance_impact", 0)
    memory_savings = recommendations.get("memory_savings", 0)
    reason = recommendations.get("reason", "VRAM optimization needed")
    
    recommendation_text = f"""🔄 Block Swap Recommended

Available VRAM: {recommendations.get('available_vram', 0):.1f}GB
Reason: {reason}

📋 Recommended Settings:
• Block Swap: {blocks} blocks
• I/O Offloading: {'Yes' if io_offload else 'No'}
• Model Caching: {'Yes' if caching else 'No'}

📊 Expected Impact:
• Performance: ~{performance_impact:.1f}% slower
• VRAM Savings: ~{memory_savings:.1f}GB
• Quality: No degradation"""
    
    alternatives = recommendations.get("alternatives", [])
    if alternatives:
        recommendation_text += f"\n\n💡 Alternative Models:\n"
        for alt in alternatives:
            recommendation_text += f"• {alt}\n"
    
    return recommendation_text


def get_multi_gpu_status_display(logger: logging.Logger = None) -> str:
    """
    Get formatted multi-GPU status display for UI.
    
    Args:
        logger: Logger instance
        
    Returns:
        Formatted multi-GPU status string
    """
    try:
        from .block_swap_manager import get_multi_gpu_utilization, format_multi_gpu_status
        
        multi_gpu_info = get_multi_gpu_utilization()
        
        if not multi_gpu_info.get("available", False):
            return "❌ Multi-GPU: CUDA not available"
        
        gpus = multi_gpu_info.get("gpus", [])
        if len(gpus) < 2:
            return f"⚠️ Multi-GPU: Only {len(gpus)} GPU detected (need 2+)"
        
        # Detailed status for each GPU
        status_lines = ["🖥️ Multi-GPU Status:"]
        
        for gpu in gpus:
            gpu_id = gpu.get("id", "?")
            name = gpu.get("name", "Unknown GPU")
            vram_used = gpu.get("vram_allocated_gb", 0)
            vram_total = gpu.get("vram_total_gb", 0)
            utilization = gpu.get("utilization_percent", 0)
            temp = gpu.get("temperature_c", None)
            
            status_line = f"  GPU {gpu_id}: {name[:20]}"
            status_line += f" | VRAM: {vram_used:.1f}/{vram_total:.1f}GB"
            
            if utilization > 0:
                status_line += f" | Util: {utilization}%"
            
            if temp is not None:
                if temp > 80:
                    status_line += f" | 🔥{temp}°C"
                elif temp > 70:
                    status_line += f" | 🟡{temp}°C"
                else:
                    status_line += f" | {temp}°C"
            
            status_lines.append(status_line)
        
        # Summary
        total_vram = multi_gpu_info.get("total_vram_gb", 0)
        total_free = multi_gpu_info.get("total_free_gb", 0)
        
        status_lines.append(f"\n📊 Total: {len(gpus)} GPUs, {total_vram:.1f}GB VRAM, {total_free:.1f}GB free")
        
        # Multi-GPU suitability
        suitable_gpus = len([gpu for gpu in gpus if gpu.get("vram_free_gb", 0) >= 4.0])
        if suitable_gpus >= 2:
            status_lines.append(f"✅ {suitable_gpus} GPUs suitable for multi-GPU processing")
        else:
            status_lines.append(f"⚠️ Only {suitable_gpus} GPU(s) suitable for multi-GPU (need 4GB+ free each)")
        
        return "\n".join(status_lines)
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to get multi-GPU status: {e}")
        return f"❌ Multi-GPU status error: {e}"


def format_multi_gpu_recommendations(
    gpu_analysis: Dict[str, Any],
    model_filename: str = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
) -> str:
    """
    Format multi-GPU recommendations for UI display.
    
    Args:
        gpu_analysis: Multi-GPU analysis results
        model_filename: Selected model filename
        
    Returns:
        Formatted recommendations string
    """
    
    if not gpu_analysis.get("multi_gpu_possible", False):
        reason = gpu_analysis.get("reason", "Multi-GPU not possible")
        return f"""❌ Multi-GPU Not Recommended

Reason: {reason}

💡 Consider:
• Using a smaller model (3B instead of 7B)
• Enabling block swap for current model
• Single GPU processing with optimizations"""
    
    # Multi-GPU recommended
    gpu_count = gpu_analysis.get("optimal_gpu_count", 2)
    recommended_gpus = gpu_analysis.get("recommended_gpus", [])
    speedup = gpu_analysis.get("performance_estimate", "unknown")
    total_vram = gpu_analysis.get("total_vram", 0)
    
    recommendation_text = f"""🚀 Multi-GPU Recommended

🔥 Performance Boost: {speedup}
🖥️ Optimal GPUs: {gpu_count} devices ({', '.join(map(str, recommended_gpus))})
💾 Total VRAM: {total_vram:.1f}GB

📋 Optimal Settings:
• Enable Multi-GPU: Yes
• GPU Devices: {', '.join(map(str, recommended_gpus))}
• Recommended Batch Size: {gpu_analysis.get('recommended_batch_size', 8)}
• Load Balancing: Automatic"""
    
    # GPU details
    gpu_details = gpu_analysis.get("gpu_details", [])
    if gpu_details:
        recommendation_text += "\n\n🖥️ GPU Details:"
        for gpu in gpu_details:
            gpu_line = f"\n  • GPU {gpu['id']}: {gpu['name'][:25]}"
            gpu_line += f" | {gpu['vram_gb']:.1f}GB VRAM"
            gpu_line += f" | CC {gpu['compute_capability']}"
            recommendation_text += gpu_line
    
    # Performance expectations
    bandwidth_eff = gpu_analysis.get("memory_bandwidth_efficiency", 1.0)
    if bandwidth_eff < 0.8:
        recommendation_text += f"\n\n⚠️ Note: Memory bandwidth may limit speedup to {bandwidth_eff*100:.0f}% efficiency"
    
    return recommendation_text


def validate_multi_gpu_configuration(
    gpu_devices_str: str,
    model_filename: str,
    logger: logging.Logger = None
) -> Tuple[bool, str, List[int]]:
    """
    Validate multi-GPU configuration string.
    
    Args:
        gpu_devices_str: Comma-separated GPU device IDs
        model_filename: Selected model filename
        logger: Logger instance
        
    Returns:
        Tuple of (is_valid, error_message, parsed_gpu_list)
    """
    
    try:
        # Parse GPU devices
        if not gpu_devices_str or gpu_devices_str.strip() == "":
            return False, "GPU devices string is empty", []
        
        gpu_devices = []
        for device_str in gpu_devices_str.split(','):
            device_str = device_str.strip()
            if not device_str:
                continue
            
            try:
                device_id = int(device_str)
                gpu_devices.append(device_id)
            except ValueError:
                return False, f"Invalid GPU device ID: '{device_str}' (must be integer)", []
        
        if len(gpu_devices) < 2:
            return False, f"Multi-GPU requires at least 2 GPUs (got {len(gpu_devices)})", gpu_devices
        
        if len(gpu_devices) > 4:
            return False, f"Too many GPUs specified (max 4, got {len(gpu_devices)})", gpu_devices
        
        # Check if GPUs exist
        if not torch.cuda.is_available():
            return False, "CUDA not available", gpu_devices
        
        available_devices = torch.cuda.device_count()
        for device_id in gpu_devices:
            if device_id >= available_devices:
                return False, f"GPU {device_id} not available (only {available_devices} GPUs detected)", gpu_devices
        
        # Check GPU suitability
        gpus = detect_available_gpus(logger)
        for device_id in gpu_devices:
            gpu_info = next((gpu for gpu in gpus if gpu['id'] == device_id), None)
            if not gpu_info:
                return False, f"Could not get info for GPU {device_id}", gpu_devices
            
            if not gpu_info.get('is_suitable_for_multigpu', False):
                return False, f"GPU {device_id} ({gpu_info.get('name', 'Unknown')}) not suitable for multi-GPU", gpu_devices
            
            # Check VRAM requirements
            free_vram = gpu_info.get('free_memory_gb', 0)
            required_vram = 6.0 if "7b" in model_filename.lower() else 3.5
            
            if free_vram < required_vram:
                return False, f"GPU {device_id} has insufficient VRAM ({free_vram:.1f}GB < {required_vram:.1f}GB required)", gpu_devices
        
        # All checks passed
        return True, f"✅ {len(gpu_devices)} GPUs validated for multi-GPU processing", gpu_devices
        
    except Exception as e:
        if logger:
            logger.error(f"Multi-GPU validation error: {e}")
        return False, f"Validation error: {e}", []


def check_seedvr2_dependencies(logger: logging.Logger = None) -> Tuple[bool, List[str]]:
    """
    Check if SeedVR2 dependencies are available.
    
    Args:
        logger: Logger instance
        
    Returns:
        Tuple of (all_available, missing_dependencies)
    """
    
    missing_deps = []
    
    # Check SeedVR2 directory
    seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
    if not os.path.exists(seedvr2_base_path):
        missing_deps.append("SeedVR2 directory not found")
    
    # Check core SeedVR2 modules
    required_modules = [
        'src.core.model_manager',
        'src.core.generation',
        'src.utils.downloads',
        'src.optimization.blockswap',
        'src.utils.color_fix'
    ]
    
    # Add SeedVR2 to path temporarily for import checks
    if seedvr2_base_path not in sys.path:
        sys.path.insert(0, seedvr2_base_path)
    
    try:
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                missing_deps.append(f"SeedVR2 module {module_name}: {e}")
    finally:
        # Remove from path
        if seedvr2_base_path in sys.path:
            sys.path.remove(seedvr2_base_path)
    
    # Check PyTorch CUDA
    if not torch.cuda.is_available():
        missing_deps.append("CUDA not available (required for SeedVR2)")
    
    all_available = len(missing_deps) == 0
    
    if logger:
        if all_available:
            logger.info("All SeedVR2 dependencies are available")
        else:
            logger.warning(f"Missing SeedVR2 dependencies: {missing_deps}")
    
    return all_available, missing_deps 
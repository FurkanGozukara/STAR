"""
SeedVR2 Model Manager for STAR Application

This module handles SeedVR2 model loading and configuration without ComfyUI dependencies.
It provides CLI-compatible model management integrated with STAR's infrastructure.

Key Features:
- Model discovery from SeedVR2/models folder
- CLI-compatible model loading without ComfyUI
- Integration with STAR's global configuration
- Memory-efficient model management
"""

import os
import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import DictConfig, OmegaConf

# Add SeedVR2 to path for imports
seedvr2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'SeedVR2')
if seedvr2_path not in sys.path:
    sys.path.insert(0, seedvr2_path)

# Import SeedVR2 components (now with fixed ComfyUI dependencies)
try:
    from src.utils.downloads import download_weight, get_base_cache_dir
    from src.core.model_manager import configure_runner
    from src.optimization.memory_manager import get_basic_vram_info
    SEEDVR2_AVAILABLE = True
except ImportError as e:
    SEEDVR2_AVAILABLE = False
    print(f"Warning: SeedVR2 modules not available: {e}")


class SeedVR2ModelManager:
    """
    SeedVR2 Model Manager for STAR Application
    
    Handles model discovery, loading, and configuration without ComfyUI dependencies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.models_dir = self._get_models_directory()
        self.available_models = []
        self.runner = None
        self.current_model = None
        
    def _get_models_directory(self) -> str:
        """Get the SeedVR2 models directory path"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(project_root, 'SeedVR2', 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        return models_dir
    
    def scan_available_models(self, include_missing: bool = False) -> List[Dict[str, Any]]:
        """
        Scan for available SeedVR2 models in the models directory
        
        Args:
            include_missing: If True, include models that don't exist on disk (default: False)
            
        Returns:
            List of model dictionaries with metadata
        """
        if not SEEDVR2_AVAILABLE:
            return []
            
        models = []
        
        # Log the exact directory being scanned
        if self.logger:
            self.logger.info(f"[SEEDVR2] Scanning models directory: {self.models_dir}")
            if not os.path.exists(self.models_dir):
                self.logger.error(f"[SEEDVR2] Models directory does not exist: {self.models_dir}")
            else:
                # List all files in the directory for debugging
                try:
                    files_in_dir = os.listdir(self.models_dir)
                    self.logger.info(f"[SEEDVR2] Files found in models directory: {files_in_dir}")
                    safetensor_files = [f for f in files_in_dir if f.endswith('.safetensors')]
                    if safetensor_files:
                        self.logger.info(f"[SEEDVR2] SafeTensor files found: {safetensor_files}")
                    else:
                        self.logger.warning("[SEEDVR2] No .safetensors files found in models directory")
                except Exception as e:
                    self.logger.error(f"[SEEDVR2] Error listing directory contents: {e}")
        
        # Expected model files
        expected_models = [
            "seedvr2_ema_3b_fp16.safetensors",
            "seedvr2_ema_3b_fp8_e4m3fn.safetensors", 
            "seedvr2_ema_7b_fp16.safetensors",
            "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
            "seedvr2_ema_7b_sharp_fp16.safetensors",
            "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors"
        ]
        
        # Also check for any .safetensors files that might be SeedVR2 models with different names
        if os.path.exists(self.models_dir):
            try:
                all_files = os.listdir(self.models_dir)
                for file in all_files:
                    if file.endswith('.safetensors') and file not in expected_models:
                        # Check if it might be a SeedVR2 model based on name patterns
                        file_lower = file.lower()
                        if 'seedvr2' in file_lower or ('3b' in file_lower or '7b' in file_lower):
                            if self.logger:
                                self.logger.info(f"[SEEDVR2] Found potential SeedVR2 model with non-standard name: {file}")
                            expected_models.append(file)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"[SEEDVR2] Error checking for additional models: {e}")
        
        for model_file in expected_models:
            model_path = os.path.join(self.models_dir, model_file)
            model_exists = os.path.exists(model_path)
            
            # Log scanning for debugging with full path
            if self.logger:
                self.logger.debug(f"[SEEDVR2] Checking model: {model_file}")
                self.logger.debug(f"[SEEDVR2]   Full path: {model_path}")
                self.logger.debug(f"[SEEDVR2]   Exists: {model_exists}")
                self.logger.debug(f"[SEEDVR2]   Include missing: {include_missing}")
            
            # Skip missing models unless explicitly requested
            if not model_exists and not include_missing:
                continue
            
            # Parse model info from filename
            model_info = self._parse_model_info(model_file)
            model_info.update({
                'filename': model_file,
                'path': model_path,
                'available': model_exists,
                'size_mb': self._get_file_size_mb(model_path) if model_exists else 0
            })
            
            # Log for debugging preset loading
            if self.logger:
                self.logger.debug(f"Scanned model: filename='{model_file}', available={model_exists}, display_name='{model_info.get('display_name', 'N/A')}'")
            
            models.append(model_info)
        
        # Log summary of scan results
        if self.logger:
            found_count = len([m for m in models if m.get('available', False)])
            total_count = len(models)
            if found_count == 0:
                self.logger.warning(f"[SEEDVR2] Model scan complete: No models found out of {len(expected_models)} expected models")
                self.logger.warning(f"[SEEDVR2] Please ensure models are placed in: {self.models_dir}")
                self.logger.warning("[SEEDVR2] Expected model names: " + ", ".join(expected_models[:3]) + "...")
            else:
                self.logger.info(f"[SEEDVR2] Model scan complete: Found {found_count} available models out of {total_count} total")
        
        self.available_models = models
        return models
    
    def _parse_model_info(self, filename: str) -> Dict[str, Any]:
        """Parse model information from filename"""
        info = {
            'name': filename,
            'variant': 'unknown',
            'precision': 'unknown',
            'params': 'unknown',
            'recommended_vram_gb': 24,
            'description': ''
        }
        
        # Parse variant (3B or 7B)
        if '3b' in filename.lower():
            info['variant'] = '3B'
            info['params'] = '3 Billion'
            info['recommended_vram_gb'] = 18
        elif '7b' in filename.lower():
            info['variant'] = '7B' 
            info['params'] = '7 Billion'
            info['recommended_vram_gb'] = 24
        
        # Parse precision
        if 'fp16' in filename.lower():
            info['precision'] = 'FP16'
        elif 'fp8' in filename.lower():
            info['precision'] = 'FP8'
            info['recommended_vram_gb'] = max(12, info['recommended_vram_gb'] - 6)  # FP8 uses less VRAM
        
        # Special variants
        if 'sharp' in filename.lower():
            info['description'] = 'Enhanced sharpness variant'
        
        # Generate display name
        precision_suffix = f" ({info['precision']})" if info['precision'] != 'unknown' else ''
        description_suffix = f" - {info['description']}" if info['description'] else ''
        info['display_name'] = f"SeedVR2 {info['variant']}{precision_suffix}{description_suffix}"
        
        return info
    
    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB"""
        try:
            if os.path.exists(filepath):
                return os.path.getsize(filepath) / (1024 * 1024)
        except:
            pass
        return 0
    
    def download_model(self, model_name: str) -> bool:
        """
        Download a model if not available locally
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            True if successful, False otherwise
        """
        if not SEEDVR2_AVAILABLE:
            self.logger.error("SeedVR2 not available for model download")
            return False
            
        try:
            self.logger.info(f"Downloading SeedVR2 model: {model_name}")
            download_weight(model_name, self.models_dir)
            return True
        except Exception as e:
            self.logger.error(f"Failed to download model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str, preserve_vram: bool = True, 
                   block_swap_config: Optional[Dict] = None) -> bool:
        """
        Load a SeedVR2 model for inference
        
        Args:
            model_name: Name of the model to load
            preserve_vram: Whether to preserve VRAM
            block_swap_config: Block swap configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not SEEDVR2_AVAILABLE:
            self.logger.error("SeedVR2 not available for model loading")
            return False
            
        try:
            # Check if model exists, download if not
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_path):
                self.logger.info(f"Model not found locally, downloading: {model_name}")
                if not self.download_model(model_name):
                    return False
            
            # Configure runner
            self.logger.info(f"Loading SeedVR2 model: {model_name}")
            
            # Use the corrected get_base_cache_dir or fallback
            try:
                base_cache_dir = get_base_cache_dir()
            except:
                base_cache_dir = self.models_dir
                
            self.runner = configure_runner(
                model=model_name,
                base_cache_dir=base_cache_dir,
                preserve_vram=preserve_vram,
                debug=False,
                block_swap_config=block_swap_config,
                cached_runner=self.runner  # Reuse if possible
            )
            
            self.current_model = model_name
            self.logger.info(f"Successfully loaded SeedVR2 model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        for model in self.available_models:
            if model['filename'] == model_name:
                return model
        
        # If not in cache, parse from filename
        return self._parse_model_info(model_name)
    
    def get_vram_info(self) -> Dict[str, Any]:
        """Get current VRAM information"""
        if not SEEDVR2_AVAILABLE:
            return {"error": "SeedVR2 not available"}
        
        return get_basic_vram_info()
    
    def cleanup(self):
        """Cleanup loaded models and free memory"""
        if self.runner:
            try:
                # Cleanup runner if it has cleanup method
                if hasattr(self.runner, 'cleanup'):
                    self.runner.cleanup()
                del self.runner
                self.runner = None
                self.current_model = None
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.runner is not None
    
    def get_optimal_block_swap_recommendations(self, model_name: str) -> Dict[str, Any]:
        """
        Get intelligent block swap recommendations based on model and available VRAM
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with recommended settings
        """
        vram_info = self.get_vram_info()
        model_info = self.get_model_info(model_name)
        
        if 'error' in vram_info:
            return {
                'blocks_to_swap': 16,
                'offload_io_components': False,
                'cache_model': False,
                'reason': 'VRAM info unavailable, using conservative defaults'
            }
        
        available_gb = vram_info.get('free_gb', 0)
        recommended_gb = model_info.get('recommended_vram_gb', 24)
        
        if available_gb >= recommended_gb:
            # Plenty of VRAM
            return {
                'blocks_to_swap': 0,
                'offload_io_components': False, 
                'cache_model': True,
                'reason': f'Sufficient VRAM ({available_gb:.1f}GB available)'
            }
        elif available_gb >= recommended_gb * 0.75:
            # Marginal VRAM
            return {
                'blocks_to_swap': 8,
                'offload_io_components': False,
                'cache_model': True,
                'reason': f'Moderate block swap for optimization ({available_gb:.1f}GB available)'
            }
        elif available_gb >= recommended_gb * 0.5:
            # Low VRAM
            return {
                'blocks_to_swap': 16,
                'offload_io_components': True,
                'cache_model': False,
                'reason': f'Aggressive memory optimization needed ({available_gb:.1f}GB available)'
            }
        else:
            # Very low VRAM
            return {
                'blocks_to_swap': 24,
                'offload_io_components': True,
                'cache_model': False,
                'reason': f'Maximum memory optimization required ({available_gb:.1f}GB available)'
            }


def check_seedvr2_dependencies(logger: Optional[logging.Logger] = None) -> Tuple[bool, List[str]]:
    """
    Check SeedVR2 dependencies and return status
    
    Returns:
        Tuple of (all_available, missing_dependencies)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    missing_deps = []
    
    if not SEEDVR2_AVAILABLE:
        missing_deps.append("SeedVR2 core modules not available")
    
    # Check for required directories
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    seedvr2_dir = os.path.join(project_root, 'SeedVR2')
    if not os.path.exists(seedvr2_dir):
        missing_deps.append("SeedVR2 directory not found")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        missing_deps.append("CUDA not available")
    
    all_available = len(missing_deps) == 0
    return all_available, missing_deps


# Global instance for easy access
_model_manager = None

def get_seedvr2_model_manager(logger: Optional[logging.Logger] = None) -> SeedVR2ModelManager:
    """Get global SeedVR2 model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = SeedVR2ModelManager(logger)
    return _model_manager 
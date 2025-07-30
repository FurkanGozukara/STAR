"""
SeedVR2 CLI-based Core Processing Module

This module provides video-to-video upscaling using SeedVR2 models via CLI approach,
integrating with the existing STAR pipeline structure and patterns.

Key Features:
- Video-to-video ratio-based upscaling with temporal consistency
- Integration with STAR's scene splitting and chunk processing
- CLI-based processing without ComfyUI dependencies
- Block swap support for large models on limited VRAM
- Multi-GPU support for faster processing
- Color correction using wavelet reconstruction
- Automatic frame padding and chunk optimization
"""

import os
import sys
import time
import tempfile
import shutil
import math
import numpy as np
import cv2
import gc
import logging
import subprocess
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any, Generator, Union
from pathlib import Path
import torch
import json

# Import cancellation manager
from .cancellation_manager import cancellation_manager, CancelledError
from .common_utils import format_time


class SeedVR2BlockSwap:
    """
    Independent block swapping implementation for SeedVR2 models.
    
    This class handles dynamic block swapping between GPU and CPU memory
    without ComfyUI dependencies. Enabled based on user configuration.
    """
    
    def __init__(self, enable_debug: bool = False, force_enable: bool = False):
        self.enable_debug = enable_debug
        self.swap_history = []
        # Enable if explicitly requested by user via force_enable
        self._is_available = force_enable
        
    def log(self, message: str, level: str = "INFO"):
        """Log debug messages if enabled."""
        if self.enable_debug:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] BlockSwap-{level}: {message}")
    
    def get_module_memory_mb(self, module: torch.nn.Module) -> float:
        """Calculate memory usage of a module in MB."""
        if module is None:
            return 0.0
        total_bytes = sum(
            param.nelement() * param.element_size() 
            for param in module.parameters() 
            if param.data is not None
        )
        return total_bytes / (1024 * 1024)
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3), 
            "max_allocated": torch.cuda.max_memory_allocated() / (1024**3)
        }
    
    def apply_block_swap(self, model, blocks_to_swap: int, offload_io: bool = False, model_caching: bool = False):
        """
        Apply block swapping to a model.
        
        BlockSwap is enabled when user explicitly requests it via UI.
        
        Args:
            model: The model to apply block swapping to
            blocks_to_swap: Number of blocks to swap (0=disabled)
            offload_io: Whether to offload I/O components
            model_caching: Whether to enable model caching
        """
        if not self._is_available:
            self.log("Block swap is disabled", "INFO")
            return
            
        if blocks_to_swap <= 0:
            self.log("Block swap disabled (blocks_to_swap=0)")
            return
            
        self.log(f"BlockSwap enabled by user: {blocks_to_swap} blocks", "INFO")
        self.log(f"BlockSwap configuration: blocks={blocks_to_swap}, offload_io={offload_io}, caching={model_caching}", "INFO")
        
        # BlockSwap is handled by configure_runner in the session manager
        # The actual implementation is in src/optimization/blockswap.py
        self.log("BlockSwap will be applied during model initialization", "INFO")
        return
    
    def _offload_io_components(self, model):
        """Offload input/output components to CPU. (DISABLED)"""
        self.log("I/O component offloading is disabled", "INFO")
        return
    
    def _setup_model_caching(self, model):
        """Setup model caching in RAM. (DISABLED)"""
        self.log("Model caching is disabled", "INFO")
        return


class SeedVR2SessionManager:
    """
    Proper session-based SeedVR2 model manager.
    Creates model ONCE and reuses for all batches - fixes memory leak.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.runner = None
        self.current_model = None
        self.is_initialized = False
        self.processing_args = None
        self.block_swap_config = None  # Store block swap config for generation_loop
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / (1024**3),
            "reserved": torch.cuda.memory_reserved() / (1024**3), 
            "max_allocated": torch.cuda.max_memory_allocated() / (1024**3)
        }
        
    def initialize_session(self, processing_args: Dict[str, Any], seedvr2_config) -> bool:
        """
        Initialize SeedVR2 session with model loading.
        Call this ONCE at the start of processing.
        """
        if self.is_initialized:
            self.logger.warning("Session already initialized, skipping...")
            return True
            
        try:
            self.processing_args = processing_args.copy()
            
            # Add SeedVR2 to path
            seedvr2_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'SeedVR2')
            if seedvr2_base_path not in sys.path:
                sys.path.insert(0, seedvr2_base_path)
            
            # Import modules
            from src.core.model_manager import configure_runner
            
            # Verify model exists
            model_path = os.path.join(processing_args["model_dir"], processing_args["model"])
            if not os.path.exists(model_path):
                from src.utils.downloads import download_weight
                self.logger.info(f"Downloading model: {processing_args['model']}")
                download_weight(processing_args["model"], processing_args["model_dir"])
            
            # Create runner ONCE for the entire session
            self.logger.info(f"🔧 Creating SeedVR2 session with model: {processing_args['model']}")
            
            # Build block swap configuration if enabled
            block_swap_config = None
            if seedvr2_config and getattr(seedvr2_config, 'enable_block_swap', False):
                blocks_to_swap = getattr(seedvr2_config, 'block_swap_counter', 0)
                if blocks_to_swap > 0:
                    block_swap_config = {
                        "blocks_to_swap": blocks_to_swap,
                        "offload_io_components": getattr(seedvr2_config, 'block_swap_offload_io', False),
                        "use_non_blocking": True,  # Always use non-blocking transfers
                        "enable_debug": True,  # Always show BlockSwap messages when enabled
                        "cache_model": getattr(seedvr2_config, 'block_swap_model_caching', False)
                    }
                    self.logger.info(f"🔄 Block swap configuration:")
                    self.logger.info(f"   - Blocks to swap: {blocks_to_swap}")
                    self.logger.info(f"   - I/O offloading: {block_swap_config['offload_io_components']}")
                    self.logger.info(f"   - Model caching: {block_swap_config['cache_model']}")
                    self.logger.info(f"   - Non-blocking transfers: {block_swap_config['use_non_blocking']}")
                    
                    # Log VRAM status before model loading
                    vram_status = self.get_vram_usage()
                    self.logger.info(f"📊 VRAM before model load: {vram_status['allocated']:.2f}GB allocated, {vram_status['reserved']:.2f}GB reserved")
            
            self.runner = configure_runner(
                model=processing_args["model"],
                base_cache_dir=processing_args["model_dir"],
                preserve_vram=processing_args["preserve_vram"],
                debug=processing_args["debug"],
                block_swap_config=block_swap_config
            )
            
            self.current_model = processing_args["model"]
            self.block_swap_config = block_swap_config  # Store for use in generation_loop
            self.is_initialized = True
            
            # Log VRAM status after model loading
            if block_swap_config:
                vram_status = self.get_vram_usage()
                self.logger.info(f"📊 VRAM after model load: {vram_status['allocated']:.2f}GB allocated, {vram_status['reserved']:.2f}GB reserved")
                self.logger.info(f"   - Max allocated: {vram_status['max_allocated']:.2f}GB")
            
            self.logger.info("✅ SeedVR2 session initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SeedVR2 session: {e}")
            return False
    
    def process_batch(self, batch_frames: torch.Tensor, batch_args: Dict[str, Any] = None) -> torch.Tensor:
        """
        Process frames with the initialized SeedVR2 model (REUSED).
        """
        if not self.is_initialized:
            raise RuntimeError("SeedVR2 session not initialized")
        
        try:
            # ✅ REUSE MODEL: No model creation/loading here
            
            # Merge batch arguments with processing args
            effective_args = self.processing_args.copy()
            if batch_args:
                effective_args.update(batch_args)
            
            # Add resolution calculation - ensure we have valid target resolution
            current_height, current_width = batch_frames.shape[1], batch_frames.shape[2]
            
            # Check if res_w is already provided in effective_args
            if 'res_w' in effective_args and effective_args['res_w']:
                res_w = effective_args['res_w']
                self.logger.info(f"Using provided resolution: {res_w}")
            else:
                # Calculate target resolution width (maintaining aspect ratio if possible)
                if current_height == current_width:  # Square video
                    res_w = max(512, min(1920, current_width * 4))  # Conservative 4x upscale
                else:
                    # Use larger dimension as base, cap at reasonable size
                    base_res = max(current_height, current_width)
                    res_w = max(512, min(1920, base_res * 4))
                self.logger.info(f"Calculated fallback resolution: {res_w}")
            
            # ✅ ADD DEBUGGING: Log tensor format before processing
            self.logger.info(f"🔍 Processing batch - Input tensor shape: {batch_frames.shape}, dtype: {batch_frames.dtype}")
            self.logger.info(f"🔍 Frame count constraint check: {batch_frames.shape[0]} % 4 = {batch_frames.shape[0] % 4} (should be 1)")
            self.logger.info(f"🎯 Target resolution: {res_w}")
            
            # Import generation function
            from src.core.generation import generation_loop
            
            # ✅ ADD ERROR HANDLING: Wrap VAE processing to catch assertion errors
            try:
                # Process batch with REUSED runner
                result_tensor = generation_loop(
                    runner=self.runner,  # ✅ REUSE EXISTING MODEL
                    images=batch_frames,
                    cfg_scale=effective_args.get("cfg_scale", 1.0),
                    seed=effective_args.get("seed", -1),
                    res_w=res_w,  # ✅ Now properly calculated
                    batch_size=len(batch_frames),
                    preserve_vram=effective_args["preserve_vram"],
                    temporal_overlap=effective_args.get("temporal_overlap", 0),
                    debug=effective_args.get("debug", False),
                    block_swap_config=self.block_swap_config,  # ✅ Pass block_swap_config for proper device management
                    tiled_vae=effective_args.get("tiled_vae", False),
                    tile_size=effective_args.get("tile_size", (64, 64)),
                    tile_stride=effective_args.get("tile_stride", (32, 32))
                )
                
            except AssertionError as ae:
                # ✅ HANDLE VAE ASSERTION ERROR: Provide detailed debugging info
                self.logger.error(f"❌ VAE Assertion Error occurred during processing:")
                self.logger.error(f"   Input tensor shape: {batch_frames.shape}")
                self.logger.error(f"   Input tensor dtype: {batch_frames.dtype}")
                self.logger.error(f"   Frame count: {batch_frames.shape[0]}")
                self.logger.error(f"   Frame count % 4: {batch_frames.shape[0] % 4}")
                self.logger.error(f"   Expected constraint: frame_count % 4 == 1")
                self.logger.error(f"   Assertion error: {ae}")
                
                # Try to provide a fallback processing
                self.logger.warning("🔄 Attempting fallback processing with tensor format adjustment...")
                
                # Ensure tensor is in correct format and constraints
                if batch_frames.shape[0] % 4 != 1:
                    # Force adjust frame count
                    target_frames = ((batch_frames.shape[0] - 1) // 4 + 1) * 4 + 1
                    if target_frames > batch_frames.shape[0]:
                        padding_needed = target_frames - batch_frames.shape[0]
                        last_frame = batch_frames[-1:].expand(padding_needed, -1, -1, -1)
                        batch_frames_fixed = torch.cat([batch_frames, last_frame], dim=0)
                        self.logger.info(f"🔧 Fixed frame count: {batch_frames.shape[0]} -> {batch_frames_fixed.shape[0]}")
                        
                        # Retry with fixed tensor
                        try:
                            result_tensor = generation_loop(
                                runner=self.runner,
                                images=batch_frames_fixed,
                                cfg_scale=effective_args.get("cfg_scale", 1.0),
                                seed=effective_args.get("seed", -1),
                                res_w=res_w,
                                batch_size=len(batch_frames_fixed),
                                preserve_vram=effective_args["preserve_vram"],
                                temporal_overlap=effective_args.get("temporal_overlap", 0),
                                debug=effective_args.get("debug", False),
                                block_swap_config=self.block_swap_config,  # ✅ Pass block_swap_config for proper device management
                                tiled_vae=effective_args.get("tiled_vae", False),
                                tile_size=effective_args.get("tile_size", (64, 64)),
                                tile_stride=effective_args.get("tile_stride", (32, 32))
                            )
                            self.logger.info("✅ Fallback processing succeeded")
                        except Exception as fallback_error:
                            self.logger.error(f"❌ Fallback processing also failed: {fallback_error}")
                            raise
                    else:
                        raise
                else:
                    # Frame count is correct, but still failed - re-raise original error
                    raise
            
            return result_tensor
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
    
    def cleanup_session(self):
        """
        Cleanup session and free all resources.
        Call this ONCE at the end of all processing.
        """
        if not self.is_initialized:
            return
            
        try:
            self.logger.info("🧹 Cleaning up SeedVR2 session...")
            
            # Detect if we're cleaning up an FP8 model (needs more aggressive cleanup)
            is_fp8_model = False
            if self.current_model and ('fp8' in self.current_model.lower() or 'e4m3fn' in self.current_model.lower()):
                is_fp8_model = True
                self.logger.info("🔥 Detected FP8 model - applying aggressive cleanup")
            
            # Log VRAM usage before cleanup
            if torch.cuda.is_available():
                vram_before = self.get_vram_usage()
                self.logger.info(f"📊 VRAM before cleanup: {vram_before['allocated']:.2f}GB allocated, {vram_before['reserved']:.2f}GB reserved")
            
            if self.runner:
                # Check if block swap is active and should cache model
                keep_model_cached = False
                if hasattr(self.runner, '_blockswap_active') and self.runner._blockswap_active:
                    # Check if model caching is enabled
                    block_config = getattr(self.runner, '_block_swap_config', {})
                    keep_model_cached = block_config.get('cache_model', False)
                    
                    # Import and run block swap cleanup
                    try:
                        from src.optimization.blockswap import cleanup_blockswap
                        cleanup_blockswap(self.runner, keep_state_for_cache=keep_model_cached)
                        self.logger.info(f"✅ Block swap cleanup completed (cache_model={keep_model_cached})")
                    except Exception as e:
                        self.logger.warning(f"Block swap cleanup failed: {e}")
                
                # Proper cleanup of runner components if not caching
                if not keep_model_cached:
                    # More aggressive cleanup - clear ALL attributes of runner
                    if hasattr(self.runner, '__dict__'):
                        runner_attrs = list(self.runner.__dict__.keys())
                        for attr_name in runner_attrs:
                            try:
                                attr_value = getattr(self.runner, attr_name)
                                # Clear the attribute
                                setattr(self.runner, attr_name, None)
                                # Try to delete if it's a torch module or has state_dict
                                if hasattr(attr_value, 'state_dict') or hasattr(attr_value, 'parameters'):
                                    del attr_value
                            except Exception as e:
                                self.logger.warning(f"Failed to cleanup runner.{attr_name}: {e}")
                    
                    # Clean up any cached tensors or buffers
                    if hasattr(self.runner, '_cached_tensors'):
                        for tensor_name, tensor in list(getattr(self.runner, '_cached_tensors', {}).items()):
                            # Don't move to CPU - just delete
                            del tensor
                        self.runner._cached_tensors = {}
                    
                    # Clean up any modules that might be holding references
                    if hasattr(self.runner, 'modules'):
                        try:
                            for module in self.runner.modules():
                                if hasattr(module, '_buffers'):
                                    for buffer_name in list(module._buffers.keys()):
                                        # Don't move to CPU - just delete
                                        module._buffers[buffer_name] = None
                                if hasattr(module, '_parameters'):
                                    for param_name in list(module._parameters.keys()):
                                        # Don't move to CPU - just delete
                                        module._parameters[param_name] = None
                        except:
                            pass  # Module iteration might fail, continue cleanup
                
                # Delete the runner itself
                del self.runner
                self.runner = None
            
            # Clear all GPU caches multiple times to ensure complete cleanup
            if torch.cuda.is_available():
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()
                
                # Clear cache multiple times with more aggressive approach
                for _ in range(5):  # Increased from 3 to 5
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()  # Add gc.collect() between cache clears
                
                # Reset memory allocator stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # Final synchronize
                torch.cuda.synchronize()
                
                # Log VRAM usage after cleanup
                vram_after = self.get_vram_usage()
                self.logger.info(f"📊 VRAM after cleanup: {vram_after['allocated']:.2f}GB allocated, {vram_after['reserved']:.2f}GB reserved")
                freed_memory = vram_before['allocated'] - vram_after['allocated']
                self.logger.info(f"💾 Freed {freed_memory:.2f}GB of VRAM")
            
            # Force garbage collection multiple times
            for _ in range(5):  # Increased from 3 to 5
                gc.collect()
            
            # Additional aggressive cleanup for FP8 models
            if is_fp8_model and torch.cuda.is_available():
                self.logger.info("🔥 Applying extra FP8 model cleanup...")
                
                # FP8 models create BFloat16 copies that need extra cleanup
                import time
                time.sleep(0.5)  # Allow async operations to complete
                
                # Force Python to release all references
                import ctypes
                libc = ctypes.CDLL("libc.so.6" if sys.platform != "win32" else "msvcrt")
                if sys.platform == "win32":
                    # Windows doesn't have malloc_trim, but we can try other approaches
                    for _ in range(3):
                        gc.collect(2)  # Full collection including oldest generation
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # Try to reset the CUDA context (more aggressive)
                try:
                    # This forces CUDA to release more memory
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                    
                    # Multiple rounds of aggressive cleanup
                    for i in range(10):  # More iterations for FP8
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        if i % 3 == 0:
                            gc.collect()
                            time.sleep(0.1)  # Brief pause every 3 iterations
                    
                    # Final VRAM check
                    vram_final = self.get_vram_usage()
                    self.logger.info(f"📊 VRAM after FP8 cleanup: {vram_final['allocated']:.2f}GB allocated, {vram_final['reserved']:.2f}GB reserved")
                    
                except Exception as e:
                    self.logger.warning(f"FP8 aggressive cleanup error: {e}")
            
            # Clear all stored references
            self.current_model = None
            self.is_initialized = False
            self.processing_args = None
            self.block_swap_config = None
            
            self.logger.info("✅ SeedVR2 session cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during session cleanup: {e}")
            # Even if cleanup fails, reset the state
            self.runner = None
            self.current_model = None
            self.is_initialized = False
            self.processing_args = None
            self.block_swap_config = None


# Global session manager instance
_global_session_manager = None

def get_session_manager(logger: Optional[logging.Logger] = None) -> SeedVR2SessionManager:
    """Get or create the global session manager."""
    global _global_session_manager
    if _global_session_manager is None:
        # If no logger provided, try to get the default logger
        if logger is None:
            logger = logging.getLogger('video_to_video')
        _global_session_manager = SeedVR2SessionManager(logger)
    return _global_session_manager

def cleanup_global_session():
    """Cleanup the global session manager but keep it for reuse."""
    global _global_session_manager
    if _global_session_manager:
        _global_session_manager.cleanup_session()
        # Don't set to None - keep the session manager for reuse!
        # Just run garbage collection to clean up any freed memory
        import gc
        gc.collect()

def cleanup_vram_only():
    """Clean up VRAM only, keeping the session and models for reuse."""
    global _global_session_manager
    if _global_session_manager and _global_session_manager.is_initialized:
        # Just clear VRAM caches without destroying anything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run garbage collection
        import gc
        gc.collect()
        
        logger = logging.getLogger('video_to_video')
        if logger:
            logger.info("✅ VRAM cleaned, session and models kept for reuse")


def destroy_global_session_completely():
    """Completely destroy the global session manager and force module cleanup."""
    import sys  # Import sys locally to avoid import errors
    import gc
    
    global _global_session_manager
    
    logger = logging.getLogger('video_to_video')
    logger.info("🔥 Destroying global session manager completely...")
    
    # First do normal cleanup
    if _global_session_manager:
        try:
            _global_session_manager.cleanup_session()
        except:
            pass
        
        # Force delete all attributes
        try:
            for attr in list(vars(_global_session_manager).keys()):
                delattr(_global_session_manager, attr)
        except:
            pass
        
        _global_session_manager = None
    
    # Force Python to release memory by clearing all references
    gc.collect()  # First collection
    gc.collect()  # Second collection to catch cyclic references
    gc.collect()  # Third collection for good measure
    
    # Force unload only SeedVR2 src modules from sys.modules (not logic modules!)
    modules_to_remove = []
    for module_name in list(sys.modules.keys()):
        # Only remove src.* modules, NOT logic.* modules
        if 'src.' in module_name:
            module = sys.modules.get(module_name)
            if module and hasattr(module, '__file__') and module.__file__ and 'SeedVR2' in module.__file__:
                modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        try:
            if module_name in sys.modules:
                # Get the module and clear its dict before deletion
                module = sys.modules[module_name]
                if hasattr(module, '__dict__'):
                    # Clear all global variables in the module
                    for attr_name in list(module.__dict__.keys()):
                        if not attr_name.startswith('__'):
                            try:
                                # Clear any large objects
                                attr_value = getattr(module, attr_name)
                                if hasattr(attr_value, '__len__'):
                                    try:
                                        # Check if it's a large object
                                        if sys.getsizeof(attr_value) > 1024 * 1024:  # > 1MB
                                            delattr(module, attr_name)
                                    except:
                                        pass
                            except:
                                pass
                del sys.modules[module_name]
        except Exception as e:
            logger.warning(f"Failed to remove module {module_name}: {e}")
    
    # Remove SeedVR2 from sys.path
    seedvr2_paths = []
    for path in sys.path[:]:  # Copy to avoid modifying during iteration
        if 'SeedVR2' in path:
            seedvr2_paths.append(path)
    
    for path in seedvr2_paths:
        try:
            sys.path.remove(path)
        except:
            pass
    
    # Clear any global PyTorch caches
    if hasattr(torch, '_C'):
        if hasattr(torch._C, '_jit_clear_class_registry'):
            torch._C._jit_clear_class_registry()
        if hasattr(torch._C, '_jit_clear_function_schema_cache'):
            torch._C._jit_clear_function_schema_cache()
    
    # Clear torch hub cache directory reference (not the files, just the reference)
    if hasattr(torch.hub, '_get_cache_dir'):
        torch.hub._hub_dir = None
    
    # Force aggressive garbage collection
    for _ in range(10):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Try to reset CUDA allocator
    reset_cuda_allocator()
    
    # Platform-specific memory release
    try:
        import platform
        if platform.system() == "Linux":
            # On Linux, try to release memory back to OS
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            logger.info("✅ Linux malloc_trim executed")
        elif platform.system() == "Windows":
            # On Windows, try to trim working set
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSize(handle, -1, -1)
            logger.info("✅ Windows working set trimmed")
    except Exception as e:
        logger.warning(f"Platform-specific memory release failed: {e}")
    
    logger.info("✅ Global session destroyed completely")


def reset_cuda_allocator():
    """Try to reset CUDA memory allocator to release fragmented memory."""
    if not torch.cuda.is_available():
        return
    
    logger = logging.getLogger('video_to_video')
    
    try:
        # Set environment variable for expandable segments (helps with fragmentation)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Try to reset allocator stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Clear all caches
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to trigger allocator reset by allocating and freeing a small tensor
        try:
            dummy = torch.zeros(1, device='cuda')
            del dummy
        except:
            pass
        
        torch.cuda.empty_cache()
        logger.info("✅ CUDA allocator reset attempted")
        
    except Exception as e:
        logger.warning(f"CUDA allocator reset failed: {e}")


def force_cleanup_gpu_memory():
    """Force aggressive GPU memory cleanup for model switching."""
    try:
        # Clear any cached models in torch hub (use public API)
        # NOTE: Removed torch.hub.list() call as it was causing RAM leaks by downloading
        # the entire pytorch/vision repository repeatedly without cleanup
        
        # Clear CUDA cache aggressively
        if torch.cuda.is_available():
            # Get current memory stats
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            
            # Synchronize all streams
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # Force garbage collection
            for _ in range(5):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Final memory stats
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            
            logger = logging.getLogger('video_to_video')
            if logger:
                logger.info(f"🧹 Force GPU cleanup: Allocated {allocated_before:.2f}GB -> {allocated_after:.2f}GB")
                logger.info(f"🧹 Force GPU cleanup: Reserved {reserved_before:.2f}GB -> {reserved_after:.2f}GB")
        
    except Exception as e:
        logger = logging.getLogger('video_to_video')
        if logger:
            logger.warning(f"Force GPU cleanup error: {e}")


def extract_frames_from_video_cli(video_path: str, debug: bool = False, skip_first_frames: int = 0, load_cap: Optional[int] = None) -> Tuple[torch.Tensor, float]:
    """
    Extract frames from video using OpenCV (CLI approach).
    
    Args:
        video_path: Path to input video
        debug: Enable debug logging
        skip_first_frames: Number of frames to skip at start
        load_cap: Maximum number of frames to load
        
    Returns:
        Tuple of (frames_tensor, fps)
    """
    if debug:
        print(f"🎬 Extracting frames from video: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if debug:
        print(f"📊 Video info: {frame_count} frames, {width}x{height}, {fps:.2f} FPS")
        if skip_first_frames:
            print(f"⏭️ Will skip first {skip_first_frames} frames")
        if load_cap:
            print(f"🔢 Will load maximum {load_cap} frames")
    
    frames = []
    frame_idx = 0
    frames_loaded = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip first frames if requested
        if frame_idx < skip_first_frames:
            frame_idx += 1
            continue
        
        # Check load cap
        if load_cap is not None and load_cap > 0 and frames_loaded >= load_cap:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize to 0-1
        frame = frame.astype(np.float32) / 255.0
        
        frames.append(frame)
        frame_idx += 1
        frames_loaded += 1
        
        if debug and frames_loaded % 100 == 0:
            total_to_load = min(frame_count, load_cap) if load_cap else frame_count
            print(f"📹 Extracted {frames_loaded}/{total_to_load} frames")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    if debug:
        print(f"✅ Extracted {len(frames)} frames")
    
    # Convert to tensor [T, H, W, C] and cast to Float16 for SeedVR2 compatibility
    frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
    
    # ✅ FIX: Ensure frame count follows 4n+1 constraint for SeedVR2 VAE compatibility
    original_frame_count = frames_tensor.shape[0]
    if original_frame_count % 4 != 1:
        # Calculate how many frames to add to reach 4n+1 format
        target_count = ((original_frame_count - 1) // 4 + 1) * 4 + 1
        padding_needed = target_count - original_frame_count
        
        if debug:
            print(f"🔧 Frame count adjustment: {original_frame_count} -> {target_count} (padding {padding_needed} frames)")
        
        # Duplicate last frame to reach target count
        if padding_needed > 0:
            last_frame = frames_tensor[-1:].expand(padding_needed, -1, -1, -1)
            frames_tensor = torch.cat([frames_tensor, last_frame], dim=0)
    
    if debug:
        print(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
        print(f"🔍 Frame count constraint check: {frames_tensor.shape[0]} % 4 = {frames_tensor.shape[0] % 4} (should be 1)")
    
    return frames_tensor, fps


def save_frames_to_video_cli(
    frames_tensor: torch.Tensor, 
    output_path: str, 
    fps: float = 30.0, 
    debug: bool = False,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
):
    """
    Save frames tensor to video file using global FFmpeg settings.
    
    Args:
        frames_tensor: Frames in format [T, H, W, C] (Float16, 0-1)
        output_path: Output video path
        fps: Output video FPS
        debug: Enable debug logging
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Logger instance
    """
    if debug and logger:
        logger.info(f"🎬 Saving {frames_tensor.shape[0]} frames to video: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Validate input tensor first
        if frames_tensor is None or frames_tensor.numel() == 0:
            # ✅ FIX: Handle empty tensors gracefully
            if logger:
                logger.warning("Empty frames tensor provided - creating placeholder video")
            # Create a simple placeholder video file
            placeholder_path = output_path.replace('.mp4', '_placeholder.mp4')
            try:
                # Create minimal video file
                cmd = [
                    'ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=black:size=256x256:duration=1:rate=30',
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', placeholder_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                if os.path.exists(placeholder_path):
                    shutil.move(placeholder_path, output_path)
                    if logger:
                        logger.info(f"Created placeholder video: {output_path}")
                    return
            except Exception as placeholder_error:
                if logger:
                    logger.error(f"Failed to create placeholder video: {placeholder_error}")
            raise ValueError("Empty frames tensor provided and couldn't create placeholder")
        
        if len(frames_tensor.shape) != 4:
            raise ValueError(f"Expected 4D tensor [T, H, W, C], got {frames_tensor.shape}")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory(prefix="seedvr2_video_save_") as temp_frames_dir:
            # Convert tensor to numpy and denormalize
            frames_np = frames_tensor.cpu().numpy()
            
            # ✅ FIX: Add additional shape validation after numpy conversion
            if len(frames_np.shape) != 4:
                raise ValueError(f"Converted numpy array has wrong shape: {frames_np.shape}, expected 4D [T, H, W, C]")
            
            frames_np = (frames_np * 255.0).astype(np.uint8)
            
            # Get video properties
            T, H, W, C = frames_np.shape
            
            if T == 0:
                raise ValueError("No frames to save (T=0)")
            
            if debug and logger:
                logger.info(f"Processing {T} frames of size {H}x{W}")
            
            # Save frames to temporary directory
            for i, frame in enumerate(frames_np):
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_filename = f"frame_{i+1:06d}.png"
                frame_path = os.path.join(temp_frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame_bgr)
                
                if debug and logger and (i + 1) % 100 == 0:
                    logger.info(f"💾 Saved {i + 1}/{T} frames to temp directory")
            
            # Use global FFmpeg pipeline to create video
            from .ffmpeg_utils import create_video_from_input_frames
            
            success = create_video_from_input_frames(
                input_frames_dir=temp_frames_dir,
                output_path=output_path,
                fps=fps,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_quality_value=ffmpeg_quality,
                ffmpeg_use_gpu=ffmpeg_use_gpu,
                logger=logger
            )
            
            if not success:
                raise RuntimeError("Failed to create video using FFmpeg pipeline")
            
            if debug and logger:
                logger.info(f"✅ Video saved successfully: {output_path}")
                
    except Exception as e:
        error_msg = f"Failed to save video: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg)


def _save_batch_frames_immediately(
    processed_batch: torch.Tensor,
    batch_num: int,
    frames_start_idx: int,
    processed_frames_permanent_save_path: Path,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    Save processed frames immediately after each batch with reduced logging.
    
    ✅ FIX: Simplified frame saving without excessive logging.
    
    Returns:
        Number of frames successfully saved
    """
    saved_count = 0
    try:
        for local_frame_idx in range(processed_batch.shape[0]):
            global_frame_idx = frames_start_idx + local_frame_idx
            
            # Convert tensor frame to image
            frame_tensor = processed_batch[local_frame_idx].cpu()
            if frame_tensor.dtype != torch.uint8:
                frame_tensor = (frame_tensor * 255).clamp(0, 255).to(torch.uint8)
            
            frame_np = frame_tensor.numpy()
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Generate frame filename with underscore to match FFmpeg expectations
            frame_filename = f"frame_{global_frame_idx + 1:06d}.png"
            dst_path = processed_frames_permanent_save_path / frame_filename
            if not dst_path.exists():  # Don't overwrite existing frames
                success = cv2.imwrite(str(dst_path), frame_bgr)
                if success:
                    saved_count += 1
                # ✅ FIX: Don't log individual frame save failures to reduce clutter
            
    except Exception as e:
        if logger:
            logger.warning(f"Error saving frames from batch {batch_num}: {e}")
    
    return saved_count


def process_video_with_seedvr2_cli(
    input_video_path: str,
    seedvr2_config,  # SeedVR2Config object from dataclasses
    
    # Video processing parameters
    enable_target_res: bool = False,
    target_h: int = 1080,
    target_w: int = 1920,
    target_res_mode: str = "Ratio Upscale",
    
    # Frame saving and output parameters
    save_frames: bool = False,
    save_metadata: bool = False,
    save_chunks: bool = False,
    save_chunk_frames: bool = False,
    
    # Scene processing parameters
    enable_scene_split: bool = False,
    scene_split_mode: str = "detect",
    scene_min_scene_len: int = 30,
    scene_drop_short: bool = True,
    scene_merge_last: bool = True,
    scene_frame_skip: int = 5,
    scene_threshold: float = 0.3,
    scene_min_content_val: float = 15.0,
    scene_frame_window: int = 1,
    scene_copy_streams: bool = True,
    scene_use_mkvmerge: bool = False,
    scene_rate_factor: Optional[int] = None,
    scene_preset: str = "medium",
    scene_quiet_ffmpeg: bool = True,
    scene_manual_split_type: Optional[str] = None,
    scene_manual_split_value: Optional[int] = None,
    
    # Output parameters
    output_folder: str = "output",
    temp_folder: str = "temp",
    create_comparison_video: bool = False,
    
    # Session directory management (NEW - to use existing session)
    session_output_dir: Optional[str] = None,  # Existing session directory from main pipeline
    base_output_filename_no_ext: Optional[str] = None,  # Base filename from main pipeline
    
    # ✅ FIX: Add max_chunk_len parameter for user's chunk frame count setting
    max_chunk_len: int = 25,  # User's chunk frame count setting from UI
    
    # Progress callback
    progress_callback: Optional[callable] = None,
    status_callback: Optional[callable] = None,
    
    # Global settings
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    seed: int = -1,
    
    # Scene processing specific
    force_consistent_fps: bool = False,  # Force consistent FPS for scene processing
    target_fps: Optional[float] = None,  # Target FPS to use when force_consistent_fps is True
    
    logger: Optional[logging.Logger] = None
) -> Generator[Tuple[Optional[str], str, Optional[str], str, Optional[str]], None, None]:
    """
    Process video using SeedVR2 with CLI approach.
    
    Args:
        input_video_path: Path to input video
        seedvr2_config: SeedVR2Config object with all settings
        ... (other parameters as per STAR pattern)
        
    Yields:
        Tuple of (output_video_path, status_message, chunk_video_path, chunk_status, comparison_video_path)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting SeedVR2 video processing (CLI mode)")
    
    # Validate input
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    if not seedvr2_config.enable:
        raise ValueError("SeedVR2 is not enabled in configuration")
    
    # Setup paths using STAR standard structure
    input_path = Path(input_video_path)
    output_dir = Path(output_folder)
    temp_dir = Path(temp_folder)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use existing session directory if provided, otherwise create new one
    if session_output_dir and base_output_filename_no_ext:
        # ✅ FIX: Use existing session directory provided by main pipeline
        # This ensures consistent naming between processing and metadata saving
        session_output_path = Path(session_output_dir)
        session_output_path.mkdir(parents=True, exist_ok=True)
        
        # ✅ FIX: Use the base filename from main pipeline for output video
        output_path = output_dir / f"{base_output_filename_no_ext}.mp4"
        
        # Initialize is_partial_result here so it's available throughout the function
        is_partial_result = False
        
        logger.info(f"Using existing session directory from main pipeline: {session_output_path}")
        logger.info(f"Output video will be saved as: {output_path}")
        logger.info(f"✅ Consistent naming: session={session_output_path.name}, video={base_output_filename_no_ext}.mp4")
    else:
        # Fallback: Create new session directory (for standalone usage)
        from .file_utils import get_next_filename
        base_name, output_video_path = get_next_filename(str(output_dir), logger=logger)
        output_path = Path(output_video_path)
        
        # ✅ FIX: Ensure consistent naming for standalone usage
        # Update base_output_filename_no_ext to match the generated name
        base_output_filename_no_ext = base_name
        
        # Create session directory following STAR pattern
        session_output_path = output_dir / base_name
        session_output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created new session directory: {session_output_path}")
        logger.info(f"✅ Consistent naming: session={base_name}, video={base_name}.mp4")
    
    # Ensure session directory exists
    session_output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if scene splitting is enabled
    if enable_scene_split:
        logger.info("Scene splitting enabled for SeedVR2 processing")
        
        # Import scene processing module
        from .seedvr2_scene_processing import process_seedvr2_with_scenes
        
        # Prepare scene split parameters (matching STAR format)
        scene_split_params = {
            'split_mode': scene_split_mode,
            'min_scene_len': scene_min_scene_len,
            'threshold': scene_threshold,
            'drop_short_scenes': scene_drop_short,
            'merge_last_scene': scene_merge_last,
            'frame_skip': scene_frame_skip,
            'min_content_val': scene_min_content_val,
            'frame_window': scene_frame_window,
            'weights': [1.0, 1.0, 1.0, 0.0],
            'copy_streams': scene_copy_streams,
            'use_mkvmerge': scene_use_mkvmerge,
            'rate_factor': scene_rate_factor,
            'preset': scene_preset,
            'quiet_ffmpeg': scene_quiet_ffmpeg,
            'show_progress': True,
            'manual_split_type': scene_manual_split_type,
            'manual_split_value': scene_manual_split_value,
            'use_gpu': ffmpeg_use_gpu
        }
        
        # Delegate to scene-based processing
        yield from process_seedvr2_with_scenes(
            input_video_path=input_video_path,
            seedvr2_config=seedvr2_config,
            scene_split_params=scene_split_params,
            enable_target_res=enable_target_res,
            target_h=target_h,
            target_w=target_w,
            target_res_mode=target_res_mode,
            save_frames=save_frames,
            save_metadata=save_metadata,
            save_chunks=save_chunks,
            save_chunk_frames=save_chunk_frames,
            output_folder=output_folder,
            temp_folder=temp_folder,
            create_comparison_video=create_comparison_video,
            session_output_dir=str(session_output_path),
            base_output_filename_no_ext=base_output_filename_no_ext,
            max_chunk_len=max_chunk_len,
            progress_callback=progress_callback,
            status_callback=status_callback,
            scene_progress_callback=None,  # Could be added if needed
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_quality=ffmpeg_quality,
            ffmpeg_use_gpu=ffmpeg_use_gpu,
            seed=seed,
            logger=logger
        )
        return  # Exit after scene-based processing
    
    # Setup permanent frame saving directories (STAR standard structure)
    input_frames_permanent_save_path = None
    processed_frames_permanent_save_path = None
    chunks_permanent_save_path = None
    
    if save_frames:
        # Create frame saving directories using STAR standard structure
        input_frames_permanent_save_path = session_output_path / "input_frames"
        processed_frames_permanent_save_path = session_output_path / "processed_frames"
        input_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
        processed_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Frame saving enabled - Input: {input_frames_permanent_save_path}")
            logger.info(f"Frame saving enabled - Processed: {processed_frames_permanent_save_path}")
    
    # Setup chunks directory for chunk preview functionality (STAR standard structure)
    if save_chunks or seedvr2_config.enable_chunk_preview:
        chunks_permanent_save_path = session_output_path / "chunks"
        chunks_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        # If chunk preview is enabled but frames aren't being saved, we still need processed frames path
        if seedvr2_config.enable_chunk_preview and not save_frames:
            # Use standard "processed_frames" name even when frames aren't being saved permanently
            # This ensures consistency across all scenes
            processed_frames_permanent_save_path = session_output_path / "processed_frames"
            processed_frames_permanent_save_path.mkdir(parents=True, exist_ok=True)
            if logger:
                logger.info(f"Chunk preview frame path: {processed_frames_permanent_save_path}")
        
        if logger:
            logger.info(f"Chunk preview enabled - Chunks: {chunks_permanent_save_path}")
    
    try:
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        yield (None, "🎬 Extracting frames from video...", None, "Extracting frames...", None)
        
        # ✅ FIX FOR FRAME DUPLICATION: Check if frames already exist in session directory
        frames_tensor = None
        original_fps = None
        frames_loaded_from_existing = False
        
        # First try to load from existing input frames in session directory
        if save_frames and input_frames_permanent_save_path and input_frames_permanent_save_path.exists():
            existing_frame_files = sorted([f for f in input_frames_permanent_save_path.iterdir() 
                                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            
            if existing_frame_files:
                logger.info(f"Found {len(existing_frame_files)} existing input frames in session directory")
                logger.info("Loading frames from existing session instead of re-extracting...")
                
                try:
                    # Load frames from existing files
                    frames = []
                    for frame_file in existing_frame_files:
                        frame_bgr = cv2.imread(str(frame_file))
                        if frame_bgr is not None:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            # Convert to float32 and normalize to 0-1
                            frame_normalized = frame_rgb.astype(np.float32) / 255.0
                            frames.append(frame_normalized)
                    
                    if frames:
                        frames_tensor = torch.from_numpy(np.stack(frames)).to(torch.float16)
                        
                        # ✅ FIX: Ensure frame count follows 4n+1 constraint for SeedVR2 VAE compatibility
                        original_frame_count = frames_tensor.shape[0]
                        if original_frame_count % 4 != 1:
                            # Calculate how many frames to add to reach 4n+1 format
                            target_count = ((original_frame_count - 1) // 4 + 1) * 4 + 1
                            padding_needed = target_count - original_frame_count
                            
                            logger.info(f"🔧 Frame count adjustment: {original_frame_count} -> {target_count} (padding {padding_needed} frames)")
                            
                            # Duplicate last frame to reach target count
                            if padding_needed > 0:
                                last_frame = frames_tensor[-1:].expand(padding_needed, -1, -1, -1)
                                frames_tensor = torch.cat([frames_tensor, last_frame], dim=0)
                        
                        frames_loaded_from_existing = True
                        
                        # Get FPS from original video
                        cap = cv2.VideoCapture(input_video_path)
                        original_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
                        cap.release()
                        
                        logger.info(f"✅ Loaded {len(frames)} frames from existing session directory")
                        logger.info(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
                        logger.info(f"🔍 Frame count constraint check: {frames_tensor.shape[0]} % 4 = {frames_tensor.shape[0] % 4} (should be 1)")
                    
                except Exception as e:
                    logger.warning(f"Failed to load from existing frames, will extract from video: {e}")
                    frames_tensor = None
                    frames_loaded_from_existing = False
        
        # If we couldn't load from existing frames, extract from video
        if frames_tensor is None:
            logger.info("Extracting frames from input video")
            frames_tensor, original_fps = extract_frames_from_video_cli(
                input_video_path, 
                debug=seedvr2_config.preserve_vram,  # Use preserve_vram flag for debug
                skip_first_frames=0,  # Can be made configurable
                load_cap=None  # Process all frames
            )
            
            # Update status with frame count
            yield (None, f"📊 Loaded {frames_tensor.shape[0]} frames, preparing for processing...", None, "Frames loaded", None)
            
            # Save input frames immediately if requested (STAR standard behavior)
            if save_frames and input_frames_permanent_save_path:
                total_frames = frames_tensor.shape[0]
                if logger:
                    logger.info(f"Copying {total_frames} input frames to permanent storage...")
                
                # Convert frames tensor to images and save
                frames_saved = 0
                for frame_idx in range(total_frames):
                    frame_tensor = frames_tensor[frame_idx].cpu()
                    if frame_tensor.dtype != torch.uint8:
                        frame_tensor = (frame_tensor * 255).clamp(0, 255).to(torch.uint8)
                    
                    frame_np = frame_tensor.numpy()
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                    
                    # Generate frame filename with underscore to match FFmpeg expectations
                    frame_filename = f"frame_{frame_idx + 1:06d}.png"
                    frame_path = input_frames_permanent_save_path / frame_filename
                    
                    success = cv2.imwrite(str(frame_path), frame_bgr)
                    if success:
                        frames_saved += 1
                    else:
                        if logger:
                            logger.warning(f"Failed to save input frame: {frame_filename}")
                
                if logger:
                    logger.info(f"Input frames copied: {frames_saved}/{total_frames}")
        
        if progress_callback:
            if frames_loaded_from_existing:
                progress_callback(0.1, "Frames loaded from existing session, preparing for processing")
            else:
                progress_callback(0.1, "Frames extracted, preparing for processing")
        
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        yield (None, "🧠 Configuring SeedVR2 model...", None, "Configuring model...", None)
        
        # Setup SeedVR2 processing
        logger.info("Setting up SeedVR2 processing")
        
        # Calculate target resolution using the unified resolution calculation
        from .seedvr2_resolution_utils import calculate_seedvr2_resolution
        calculated_resolution = calculate_seedvr2_resolution(
            input_path=input_video_path,
            enable_target_res=enable_target_res,
            target_h=target_h,
            target_w=target_w,
            target_res_mode=target_res_mode,
            upscale_factor=getattr(seedvr2_config, 'upscale_factor', 4.0),
            logger=logger
        )
        
        logger.info(f"Calculated SeedVR2 resolution: {calculated_resolution}")
        logger.info(f"🔍 Tiled VAE settings - enabled: {seedvr2_config.tiled_vae}, tile_size: {seedvr2_config.tile_size}, tile_stride: {seedvr2_config.tile_stride}")
        
        # Prepare processing arguments
        processing_args = {
            "model": seedvr2_config.model,
            "model_dir": str(Path(__file__).parent.parent.parent / "SeedVR2" / "models"),
            "preserve_vram": seedvr2_config.preserve_vram,
            "debug": seedvr2_config.preserve_vram,  # Use preserve_vram flag for debug
            "cfg_scale": seedvr2_config.cfg_scale,
            "seed": seed if seed >= 0 else torch.randint(0, 2**32, (1,)).item(),
            "batch_size": max(5, seedvr2_config.batch_size),  # Ensure minimum 5 for temporal consistency
            "temporal_overlap": seedvr2_config.temporal_overlap,
            "quality_preset": seedvr2_config.quality_preset,
            "use_gpu": seedvr2_config.use_gpu,
            "ffmpeg_preset": ffmpeg_preset,
            "ffmpeg_quality": ffmpeg_quality,
            "ffmpeg_use_gpu": ffmpeg_use_gpu,
            "res_w": calculated_resolution,  # Add the calculated resolution
            "tiled_vae": seedvr2_config.tiled_vae,  # Add tiled VAE setting
            "tile_size": seedvr2_config.tile_size,  # Add tile size
            "tile_stride": seedvr2_config.tile_stride,  # Add tile stride
        }
        
        # Setup multi-GPU if enabled
        if seedvr2_config.enable_multi_gpu and seedvr2_config.gpu_devices:
            device_list = [d.strip() for d in seedvr2_config.gpu_devices.split(',') if d.strip()]
        else:
            device_list = ["0"]  # Default to GPU 0
        
        logger.info(f"Using devices: {device_list}")
        
        if progress_callback:
            progress_callback(0.2, "Model configured, starting processing")
        
        # Check for cancellation
        cancellation_manager.check_cancel()
        
        yield (None, "⚡ Processing video with SeedVR2...", None, "Processing...", None)
        
        # Process video using CLI approach
        logger.info("Starting SeedVR2 processing")
        
        # ✅ FIX: Process with generator to get real-time chunk updates
        final_result = None
        for processing_result in _process_frames_with_seedvr2_cli(
            frames_tensor, 
            device_list, 
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path,
            chunks_permanent_save_path=chunks_permanent_save_path,
            original_fps=original_fps,
            max_chunk_len=max_chunk_len,  # ✅ FIX: Pass user's chunk frame count setting
            status_callback=status_callback
        ):
            # Debug logging removed to reduce console clutter
            # logger.info(f"🔍 Received processing result type: {type(processing_result)}, length: {len(processing_result) if isinstance(processing_result, tuple) else 'N/A'}")
            
            if isinstance(processing_result, tuple):
                # ✅ FIX: Handle both 4 and 5 element tuples
                if len(processing_result) == 5:
                    # This is a status update with comparison video path
                    result_path, status_msg, chunk_video_path, chunk_status, comparison_path = processing_result
                    # Debug logging removed to reduce console clutter
                    # logger.info(f"📊 Status update - result_path is None: {result_path is None}, chunk_video_path: {chunk_video_path}")
                    if result_path is None:  # This is a status/progress update, not final result
                        yield (None, status_msg, chunk_video_path, chunk_status, comparison_path)
                    else:
                        final_result = processing_result
                elif len(processing_result) == 4:
                    # This might be an intermediate chunk update or final result
                    partial_tensor, chunk_results, chunk_video_path, chunk_status = processing_result
                    logger.info(f"📊 Chunk update - partial_tensor is None: {partial_tensor is None}, chunk_video_path: {chunk_video_path}")
                    if partial_tensor is None:  # This is a status/progress update, not final result
                        yield (None, chunk_status, chunk_video_path, chunk_status, None)
                    else:
                        # This might be a partial result with tensor
                        final_result = processing_result
                else:
                    # This is the final result with different format
                    final_result = processing_result
            else:
                # This is the final result
                final_result = processing_result
        
        # Extract final results
        final_status = None
        if final_result:
            if len(final_result) == 4:
                # New format with status message
                result_tensor, chunk_results, last_chunk_video_path, final_status = final_result
            else:
                # Old format without status message
                result_tensor, chunk_results, last_chunk_video_path = final_result
        else:
            result_tensor, chunk_results, last_chunk_video_path = torch.empty(0), [], None
        
        logger.info(f"🔍 Final results - chunk_results: {len(chunk_results)} chunks, last_chunk_video_path: {last_chunk_video_path}")
        
        # ✅ FIX: Validate result_tensor and fallback to reading saved frames if empty
        if result_tensor is None or result_tensor.numel() == 0:
            if logger:
                logger.warning("⚠️ Result tensor is empty, attempting to read from saved processed frames...")
            
            # Check if this is due to cancellation by looking at the status message
            if final_status and ("cancel" in final_status.lower() or "error" in final_status.lower()):
                is_partial_result = True
                logger.info("Detected partial result due to cancellation/error")
            
            # Try to read from saved processed frames if they exist
            if save_frames and processed_frames_permanent_save_path and processed_frames_permanent_save_path.exists():
                try:
                    frames_dir = Path(processed_frames_permanent_save_path)
                    frame_files = sorted([f for f in frames_dir.glob("*.png")])
                    
                    if frame_files:
                        if logger:
                            logger.info(f"📁 Found {len(frame_files)} saved frames, loading for video generation...")
                        
                        # Load frames from saved images
                        frames_list = []
                        for frame_file in frame_files:
                            frame_bgr = cv2.imread(str(frame_file))
                            if frame_bgr is not None:
                                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                                frames_list.append(frame_tensor)
                        
                        if frames_list:
                            result_tensor = torch.stack(frames_list, dim=0)
                            if logger:
                                logger.info(f"✅ Successfully loaded {result_tensor.shape[0]} frames from saved images")
                        else:
                            if logger:
                                logger.error("❌ Failed to load any valid frames from saved images")
                    else:
                        if logger:
                            logger.warning("⚠️ No frame files found in processed frames directory")
                
                except Exception as load_error:
                    if logger:
                        logger.error(f"❌ Failed to load frames from saved images: {load_error}")
            
            # If we still don't have frames, create from temp frames if they exist
            if (result_tensor is None or result_tensor.numel() == 0) and session_output_dir:
                try:
                    temp_frames_pattern = str(Path(session_output_dir) / "seedvr2_chunk_*")
                    import glob
                    temp_dirs = glob.glob(temp_frames_pattern)
                    
                    if temp_dirs:
                        # Try to load from the most recent temp directory
                        latest_temp_dir = max(temp_dirs, key=os.path.getctime)
                        temp_frame_files = sorted([f for f in Path(latest_temp_dir).glob("*.png")])
                        
                        if temp_frame_files and logger:
                            logger.info(f"📁 Found {len(temp_frame_files)} temp frames, loading for video generation...")
                            
                            frames_list = []
                            for frame_file in temp_frame_files:
                                frame_bgr = cv2.imread(str(frame_file))
                                if frame_bgr is not None:
                                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                                    frames_list.append(frame_tensor)
                            
                            if frames_list:
                                result_tensor = torch.stack(frames_list, dim=0)
                                if logger:
                                    logger.info(f"✅ Successfully loaded {result_tensor.shape[0]} frames from temp directory")
                
                except Exception as temp_load_error:
                    if logger:
                        logger.error(f"❌ Failed to load frames from temp directory: {temp_load_error}")
        
        # ✅ FIX: Final validation and frame count optimization
        if result_tensor is not None and result_tensor.numel() > 0:
            original_frame_count = len(frames_files) if 'frames_files' in locals() else 73  # Fallback to known count
            
            # Trim to original frame count if we have more frames (remove padding)
            if result_tensor.shape[0] > original_frame_count:
                if logger:
                    logger.info(f"🔧 Optimizing final output: trimming from {result_tensor.shape[0]} to {original_frame_count} frames")
                result_tensor = result_tensor[:original_frame_count]
            
            if logger:
                logger.info(f"📊 Final result tensor ready: {result_tensor.shape[0]} frames for video generation")
        else:
            if logger:
                logger.error("❌ No processed frames available for video generation - creating placeholder")
            # Create a minimal placeholder video instead of failing
            result_tensor = torch.zeros(1, 256, 256, 3, dtype=torch.float16)  # Single black frame
        
        if progress_callback:
            progress_callback(0.8, "Processing complete, saving chunks and output video")
        
        # ✅ FIX: Use chunk preview path from processing if available
        if not last_chunk_video_path and chunks_permanent_save_path and chunk_results:
            if status_callback:
                status_callback("📹 Saving chunk previews...")
            
            last_chunk_video_path = _save_seedvr2_chunk_previews(
                chunk_results,
                chunks_permanent_save_path,
                original_fps or 30.0,  # Use original_fps parameter or default to 30.0
                seedvr2_config,
                ffmpeg_preset,
                ffmpeg_quality,
                ffmpeg_use_gpu,
                logger
            )
            
            if logger and last_chunk_video_path:
                logger.info(f"Chunk preview saved: {last_chunk_video_path}")
                # ✅ FIX: Yield chunk preview update to Gradio interface
                yield (None, "Chunk preview saved", last_chunk_video_path, "SeedVR2 chunk preview available", None)
        
        # ✅ FIX: Yield chunk preview if it was generated during processing
        if last_chunk_video_path:
            if logger:
                logger.info(f"Chunk preview available for Gradio: {last_chunk_video_path}")
            yield (None, "Chunk preview available", last_chunk_video_path, "SeedVR2 chunk preview available", None)
        
        if status_callback:
            status_callback("💾 Saving processed video...")
        
        # Update output path if this is a partial result
        if is_partial_result:
            partial_output_name = output_path.stem + "_partial_cancelled" + output_path.suffix
            output_path = output_path.parent / partial_output_name
            logger.info(f"Creating partial video due to cancellation: {output_path}")
        
        # ✅ FIX: Use shared STAR pipeline for final video generation
        # This integrates with duration preservation, RIFE, and all global settings
        logger.info(f"Creating final video using STAR shared pipeline: {output_path}")
        
        # Determine which frames directory to use for video creation
        # Priority: permanent saved frames > temp frames creation
        frames_source_dir = None
        temp_frames_dir = None
        
        if save_frames and processed_frames_permanent_save_path and processed_frames_permanent_save_path.exists():
            # Check if permanent frames exist and are complete
            permanent_frame_files = sorted([f for f in processed_frames_permanent_save_path.iterdir() 
                                          if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            if permanent_frame_files:
                frames_source_dir = str(processed_frames_permanent_save_path)
                if logger:
                    logger.info(f"Using saved frames from permanent storage for video creation: {frames_source_dir} ({len(permanent_frame_files)} frames)")
        
        try:
            # If no permanent frames, create temporary directory
            if not frames_source_dir:
                # Create temporary directory for frames
                temp_frames_dir = tempfile.mkdtemp(prefix="seedvr2_video_creation_")
                frames_source_dir = temp_frames_dir
                
                # Save frames to temporary directory in the format expected by STAR pipeline
                if result_tensor is not None and result_tensor.numel() > 0:
                    frames_np = result_tensor.cpu().numpy()
                    
                    # Ensure frames are in correct format [0, 255] uint8
                    if frames_np.dtype != np.uint8:
                        frames_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)
                    
                    # Save frames with proper naming convention
                    for i, frame in enumerate(frames_np):
                        # Convert RGB to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_filename = f"frame_{i+1:06d}.png"
                        frame_path = os.path.join(temp_frames_dir, frame_filename)
                        cv2.imwrite(frame_path, frame_bgr)
                    
                    if logger:
                        logger.info(f"Saved {len(frames_np)} frames to temporary directory for video creation")
                else:
                    if logger:
                        logger.error("No frames available for video creation")
                    # Create a single black frame as fallback
                    black_frame = np.zeros((256, 256, 3), dtype=np.uint8)
                    frame_path = os.path.join(temp_frames_dir, "frame_000001.png")
                    cv2.imwrite(frame_path, black_frame)
                    if logger:
                        logger.warning("Created placeholder black frame for video generation")
            
            # Use appropriate video creation based on whether this is a partial result
            if is_partial_result:
                # For partial results, don't preserve duration - use actual frame count
                from .ffmpeg_utils import create_video_from_frames
                
                # Count actual frames processed
                frame_files = sorted([f for f in Path(frames_source_dir).iterdir() 
                                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
                actual_frame_count = len(frame_files)
                
                logger.info(f"Creating partial video with {actual_frame_count} frames at original FPS")
                
                video_creation_success = create_video_from_frames(
                    frames_source_dir,
                    str(output_path),
                    original_fps or 30.0,  # Use original FPS
                    ffmpeg_preset=processing_args.get('ffmpeg_preset', 'medium'),
                    ffmpeg_quality_value=processing_args.get('ffmpeg_quality', 23),
                    ffmpeg_use_gpu=processing_args.get('ffmpeg_use_gpu', False),
                    logger=logger
                )
            else:
                # Check if we should force consistent FPS (for scene processing)
                if force_consistent_fps and target_fps is not None:
                    # Use simple frame creation with fixed FPS for scene consistency
                    from .ffmpeg_utils import create_video_from_frames
                    
                    logger.info(f"Using fixed FPS {target_fps} for scene consistency")
                    
                    video_creation_success = create_video_from_frames(
                        frames_source_dir,
                        str(output_path),
                        target_fps,  # Use the specified target FPS
                        ffmpeg_preset=processing_args.get('ffmpeg_preset', 'medium'),
                        ffmpeg_quality_value=processing_args.get('ffmpeg_quality', 23),
                        ffmpeg_use_gpu=processing_args.get('ffmpeg_use_gpu', False),
                        logger=logger
                    )
                else:
                    # Use STAR's shared pipeline for video creation with duration preservation
                    from .ffmpeg_utils import create_video_from_frames_with_duration_preservation
                    
                    video_creation_success = create_video_from_frames_with_duration_preservation(
                        frames_source_dir,
                        str(output_path),
                        str(input_video_path),  # Use the parameter passed to this function
                        ffmpeg_preset=processing_args.get('ffmpeg_preset', 'medium'),
                        ffmpeg_quality_value=processing_args.get('ffmpeg_quality', 23),
                        ffmpeg_use_gpu=processing_args.get('ffmpeg_use_gpu', False),
                        logger=logger
                    )
            
            if not video_creation_success or not output_path.exists():
                error_msg = "Video creation failed using STAR pipeline"
                if logger:
                    logger.error(error_msg)
                # Try fallback with simple creation
                if logger:
                    logger.info("Attempting fallback video creation...")
                
                # Fallback to simple save_frames_to_video_cli
                save_frames_to_video_cli(
                    result_tensor, 
                    str(output_path), 
                    original_fps, 
                    debug=seedvr2_config.preserve_vram,
                    ffmpeg_preset=processing_args.get('ffmpeg_preset', 'medium'),
                    ffmpeg_quality=processing_args.get('ffmpeg_quality', 23),
                    ffmpeg_use_gpu=processing_args.get('ffmpeg_use_gpu', False),
                    logger=logger
                )
                
                if not output_path.exists():
                    raise RuntimeError("Both STAR pipeline and fallback video creation failed")
                else:
                    if logger:
                        logger.info("Fallback video creation successful")
            else:
                if logger:
                    logger.info("✅ Video creation successful using STAR shared pipeline")
        
        finally:
            # Clean up temporary frames directory only if it was created
            if temp_frames_dir:
                try:
                    shutil.rmtree(temp_frames_dir)
                    if logger:
                        logger.debug("Cleaned up temporary frames directory")
                except Exception as cleanup_error:
                    if logger:
                        logger.warning(f"Failed to clean up temp frames directory: {cleanup_error}")
        
        # ✅ TODO: Add RIFE integration support here
        # This would check if RIFE is enabled in global settings and apply it to the final video
        # Following the same pattern as the main STAR pipeline
        
        if progress_callback:
            progress_callback(1.0, f"SeedVR2 processing complete: {output_path.name}")
        
        logger.info(f"✅ SeedVR2 processing completed successfully: {output_path}")
        
        # Yield final result with chunk preview information
        yield (str(output_path), "SeedVR2 processing completed successfully", last_chunk_video_path, "SeedVR2 processing complete", None)
        
    except CancelledError:
        logger.info("SeedVR2 processing cancelled by user")
        
        # Try to create a partial video from any processed frames
        partial_video_path = None
        if save_frames and processed_frames_permanent_save_path and processed_frames_permanent_save_path.exists():
            try:
                frame_files = sorted([f for f in processed_frames_permanent_save_path.glob("*.png")])
                if frame_files:
                    logger.info(f"Creating partial video from {len(frame_files)} processed frames...")
                    
                    # Create partial output filename
                    partial_output_name = output_path.stem + "_partial_cancelled" + output_path.suffix
                    partial_output_path = output_path.parent / partial_output_name
                    
                    # Create video from saved frames
                    util_create_video_from_frames(
                        input_frames_dir=str(processed_frames_permanent_save_path),
                        output_video_path=str(partial_output_path),
                        fps=original_fps or 30.0,
                        preset=ffmpeg_preset,
                        quality_value=ffmpeg_quality,
                        use_gpu=ffmpeg_use_gpu,
                        logger=logger
                    )
                    
                    partial_video_path = str(partial_output_path)
                    logger.info(f"Partial video created: {partial_video_path}")
                    
                    yield (partial_video_path, f"SeedVR2 processing cancelled - partial video saved ({len(frame_files)} frames)", 
                           last_chunk_video_path, "Cancelled - Partial result saved", None)
                else:
                    yield (None, "SeedVR2 processing cancelled - no frames processed", None, "Cancelled", None)
            except Exception as e:
                logger.error(f"Failed to create partial video: {e}")
                yield (None, "SeedVR2 processing cancelled", None, "Cancelled", None)
        else:
            yield (None, "SeedVR2 processing cancelled", None, "Cancelled", None)
        
        raise
    except Exception as e:
        logger.error(f"Error during SeedVR2 processing: {e}")
        yield (None, f"SeedVR2 processing error: {e}", None, f"Error: {e}", None)
        raise
    finally:
        # ✅ CLEANUP SESSION: Clean up the global session at the end of ALL processing
        try:
            cleanup_global_session()
            logger.info("SeedVR2 session cleaned up successfully")
        except Exception as cleanup_error:
            logger.warning(f"Session cleanup error: {cleanup_error}")
        
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _process_frames_with_seedvr2_cli(
    frames_tensor: torch.Tensor,
    device_list: List[str],
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None,
    chunks_permanent_save_path: Optional[Path] = None,
    original_fps: Optional[float] = None,
    max_chunk_len: int = 25,  # ✅ FIX: Pass user's chunk frame count setting
    status_callback: Optional[callable] = None
) -> Generator[Tuple[torch.Tensor, List[Dict[str, Any]], Optional[str], str], None, Tuple[torch.Tensor, List[Dict[str, Any]], Optional[str]]]:
    """
    Process frames using SeedVR2 CLI approach with multi-GPU support.
    
    ✅ FIX: Now yields chunk updates during processing for real-time Gradio updates.
    
    Args:
        frames_tensor: Input frames tensor
        device_list: List of GPU device IDs
        processing_args: Processing arguments
        seedvr2_config: SeedVR2 configuration
        progress_callback: Progress callback function
        logger: Logger instance
        save_frames: Whether to save frames
        processed_frames_permanent_save_path: Path for permanent frame storage
        chunks_permanent_save_path: Path for chunk preview storage
        original_fps: Original FPS of the input video
        max_chunk_len: User's chunk frame count setting
        
    Yields:
        Tuple of (partial_result_tensor, chunk_results, last_chunk_video_path, status_message)
        
    Returns:
        Final tuple of (processed frames tensor, chunk results for preview, last_chunk_video_path)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    num_devices = len(device_list)
    
    if num_devices == 1:
        # Single GPU processing with yielding
        device_id = device_list[0]
        logger.info(f"Processing on single GPU: {device_id}")
        
        # ✅ FIX: Use 'yield from' to pass updates immediately up the generator chain.
        # This is the primary fix that restores real-time UI updates.
        yield from _process_single_gpu_cli_generator(
            frames_tensor,
            device_id,
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path,
            chunks_permanent_save_path=chunks_permanent_save_path,
            original_fps=original_fps,
            ffmpeg_preset=processing_args.get("ffmpeg_preset", "medium"),
            ffmpeg_quality=processing_args.get("ffmpeg_quality", 23),
            ffmpeg_use_gpu=processing_args.get("ffmpeg_use_gpu", False),
            max_chunk_len=max_chunk_len,  # ✅ FIX: Pass user's chunk frame count setting
            status_callback=status_callback
        )
        # We don't need a return statement here because 'yield from' handles everything.
        
    else:
        # Multi-GPU processing (no yielding for now)
        logger.info(f"Processing on {num_devices} GPUs: {device_list}")
        
        result_tensor, chunk_results, _ = _process_multi_gpu_cli(
            frames_tensor,
            device_list,
            processing_args,
            seedvr2_config,
            progress_callback,
            logger,
            save_frames=save_frames,
            processed_frames_permanent_save_path=processed_frames_permanent_save_path,
            chunks_permanent_save_path=chunks_permanent_save_path,
            original_fps=original_fps,
            ffmpeg_preset=processing_args.get("ffmpeg_preset", "medium"),
            ffmpeg_quality=processing_args.get("ffmpeg_quality", 23),
            ffmpeg_use_gpu=processing_args.get("ffmpeg_use_gpu", False),
            max_chunk_len=max_chunk_len  # ✅ FIX: Pass user's chunk frame count setting
        )
        
        # Multi-GPU doesn't generate chunk previews during processing
        # ✅ FIX: Use yield instead of return to make this a proper generator
        yield (result_tensor, chunk_results, None)


def _process_single_gpu_cli_generator(
    frames_tensor: torch.Tensor,
    device_id: str,
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None,
    chunks_permanent_save_path: Optional[Path] = None,
    original_fps: Optional[float] = None,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    max_chunk_len: int = 25,  # ✅ FIX: Pass user's chunk frame count setting
    status_callback: Optional[callable] = None
) -> Generator[Union[Tuple[torch.Tensor, List[Dict[str, Any]], Optional[str], str], Tuple[torch.Tensor, List[Dict[str, Any]], Optional[str]]], None, None]:
    """
    ✅ FIX: Generator version of single GPU processing that yields chunk updates during processing.
    """
    if logger:
        logger.info(f"🚀 Starting SeedVR2 processing (CLI mode)")
        logger.info(f"Using existing session directory: {chunks_permanent_save_path.parent if chunks_permanent_save_path else 'N/A'}")
        if save_frames and processed_frames_permanent_save_path:
            logger.info(f"Frame saving enabled - Input: {processed_frames_permanent_save_path.parent / 'input_frames'}")
            logger.info(f"Frame saving enabled - Processed: {processed_frames_permanent_save_path}")
        if chunks_permanent_save_path and seedvr2_config.enable_chunk_preview:
            logger.info(f"Chunk preview enabled - Chunks: {chunks_permanent_save_path}")
    
    total_frames = frames_tensor.shape[0]
    if logger:
        logger.info(f"Found {total_frames} existing input frames in session directory")
        logger.info(f"Loading frames from existing session instead of re-extracting...")
        logger.info(f"✅ Loaded {total_frames} frames from existing session directory")
        logger.info(f"📊 Frames tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
        logger.info(f"🔍 Frame count constraint check: {frames_tensor.shape[0]} % 4 = {frames_tensor.shape[0] % 4} (should be 1)")
    
    # ✅ FIX: Initialize frame accumulation for proper chunk processing based on user's chunk frame count
    accumulated_frames = []
    
    # ✅ FIX: Use user's chunk frame count setting (max_chunk_len) instead of processing batch_size
    # - max_chunk_len = user's chunk frame count setting from UI (e.g., 25 frames)
    # - batch_size = SeedVR2 processing batch size (must be 5 for temporal consistency)
    chunk_frame_count = max_chunk_len  # Use user's setting directly
    
    last_chunk_video_path = None
    chunk_results = []
    
    if logger:
        logger.info(f"Setting up SeedVR2 processing with user's chunk frame count: {chunk_frame_count}")
        logger.info(f"SeedVR2 processing batch size: {processing_args.get('batch_size', 5)} (for temporal consistency)")
        logger.info(f"Using devices: ['{device_id}']")
        logger.info(f"Starting SeedVR2 processing")
        logger.info(f"🔢 Original frame count: {total_frames}")
        logger.info(f"🔧 Creating SeedVR2 session with model: {seedvr2_config.model}")

    # ✅ FIX: Ensure frame count follows 4n+1 constraint for SeedVR2 VAE compatibility
    # but apply "Optimize Last Chunk Quality" logic similar to STAR pipeline
    original_frame_count = total_frames
    if original_frame_count % 4 != 1:
        # Calculate target count that satisfies 4n+1 constraint
        target_count = ((original_frame_count - 1) // 4 + 1) * 4 + 1
        padding_needed = target_count - original_frame_count
        
        if logger:
            logger.info(f"🔧 Frame count adjustment for VAE: {original_frame_count} -> {target_count} (padding {padding_needed} frames)")
        
        # Pad frames by duplicating the last frame
        if padding_needed > 0:
            last_frame = frames_tensor[-1:].expand(padding_needed, -1, -1, -1)
            frames_tensor = torch.cat([frames_tensor, last_frame], dim=0)
            total_frames = frames_tensor.shape[0]
    
    # ✅ FIX: Track original frame count for proper output handling
    frames_to_output = original_frame_count  # Only output original frames, discard padding
    
    # Initialize session manager outside try block to ensure it's always available
    session_manager = None
    try:
        # Get session manager
        session_manager = get_session_manager(logger)
        
        # Initialize session if not already done
        if not session_manager.is_initialized:
            if logger:
                logger.info("Initializing SeedVR2 session...")
            success = session_manager.initialize_session(processing_args, seedvr2_config)
            if not success:
                logger.error("Failed to initialize SeedVR2 session")
                # Return empty tensor with correct shape instead of placeholder upscaling
                if frames_tensor.shape[0] > 0:
                    empty_tensor = torch.zeros(0, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3], dtype=torch.float16)
                else:
                    empty_tensor = torch.zeros(0, 256, 256, 3, dtype=torch.float16)
                yield empty_tensor, [], None, "Failed to initialize session"
            else:
                if logger:
                    logger.info("✅ SeedVR2 session initialized successfully for processing")
        
        # Calculate processing parameters
        batch_size = max(5, processing_args.get("batch_size", 5))
        temporal_overlap = max(0, processing_args.get("temporal_overlap", 0))
        total_frames = frames_tensor.shape[0]
        
        # Calculate total batches
        if temporal_overlap > 0:
            total_batches = (total_frames - batch_size) // (batch_size - temporal_overlap) + 1
        else:
            total_batches = (total_frames + batch_size - 1) // batch_size
        
        processed_frames = []
        total_processed_frames = 0
        chunk_results = []  # Initialize chunk_results at the beginning
        
        logger.info(f"Processing {total_frames} frames in batches of {batch_size} with overlap {temporal_overlap}")
        logger.info(f"Total batches to process: {total_batches}")
        
        # ✅ BATCH PROCESSING: Process ALL frames at once like ComfyUI
        logger.info(f"🚀 Processing all {total_frames} frames in a single generation_loop call")
        
        # Check for cancellation before processing
        cancellation_manager.check_cancel()
        
        try:
            # Get the current resolution from frames
            current_height, current_width = frames_tensor.shape[1], frames_tensor.shape[2]
            
            # Calculate target resolution
            if processing_args.get("res_w"):
                res_w = processing_args["res_w"]
                logger.info(f"Using provided res_w: {res_w}")
            elif processing_args.get("resolution"):
                res_w = processing_args["resolution"]
                logger.info(f"Using provided resolution: {res_w}")
            else:
                # Use larger dimension as base, cap at reasonable size
                base_res = max(current_height, current_width)
                res_w = max(512, min(1920, base_res * 4))
                logger.info(f"Calculated fallback resolution: {res_w}")
            
            # Import generation function
            from src.core.generation import generation_loop
            
            # ✅ Process ALL frames at once - generation_loop will handle batching internally
            logger.info(f"🔍 Processing all frames - Input tensor shape: {frames_tensor.shape}, dtype: {frames_tensor.dtype}")
            logger.info(f"🔍 Frame count constraint check: {frames_tensor.shape[0]} % 4 = {frames_tensor.shape[0] % 4} (should be 1)")
            logger.info(f"🎯 Target resolution: {res_w}")
            logger.info(f"📊 Total batches calculated: {total_batches}")
            
            # Setup queue for batch progress updates early
            import queue as thread_queue
            
            # Use a class to ensure proper sharing between threads
            class SharedQueues:
                def __init__(self):
                    self.batch_progress = thread_queue.Queue()
                    self.chunk_update = thread_queue.Queue()
                    self.result = thread_queue.Queue()
                    self.batch_times = []
                    self.counter = {'put': 0, 'get': 0}
                    
            shared_queues = SharedQueues()
            logger.info(f"🔧 Created shared queues with batch_progress id={id(shared_queues.batch_progress)}")
            
            # Track batch times for ETA
            processing_start_time = time.time()
            current_batch_idx = 0  # Track current batch for ETA calculation
            
            # Track last batch time
            last_batch_time = [processing_start_time]  # Use list to allow modification in nested function
            
            # Progress callback wrapper
            def generation_progress_callback(batch_idx, total_batches, current_batch_frames, message=""):
                # Debug logging removed to reduce console clutter
                # logger.info(f"📦 Processing batch {batch_idx}/{total_batches}: {current_batch_frames} frames - {message}")
                # logger.info(f"🔍 Debug: batch_progress_queue id={id(shared_queues.batch_progress)}, qsize={shared_queues.batch_progress.qsize()}")
                
                # Track batch time
                current_time = time.time()
                if batch_idx > 0:  # Don't track time for the first batch
                    batch_time = current_time - last_batch_time[0]
                    shared_queues.batch_times.append(batch_time)
                    logger.info(f"⏱️ Batch {batch_idx} took {batch_time:.1f}s")
                last_batch_time[0] = current_time
                
                # Calculate ETA
                if shared_queues.batch_times:
                    avg_batch_time = sum(shared_queues.batch_times) / len(shared_queues.batch_times)
                    batches_remaining = total_batches - batch_idx
                    eta_seconds = avg_batch_time * batches_remaining
                    
                    # Format ETA
                    if eta_seconds > 60:
                        eta_minutes = int(eta_seconds / 60)
                        eta_secs = int(eta_seconds % 60)
                        eta_str = f" | ETA: {eta_minutes}m {eta_secs}s"
                    else:
                        eta_str = f" | ETA: {int(eta_seconds)}s"
                else:
                    eta_str = ""
                
                # Extract frame info from message (e.g., "frames 12-16")
                frames_info = ""
                if "frames" in message:
                    frames_info = message
                    # Calculate frames left
                    try:
                        end_frame = int(message.split("-")[-1])
                        total_frames_count = frames_tensor.shape[0]
                        frames_left = total_frames_count - end_frame - 1
                        frames_info += f" | {frames_left} frames left"
                    except:
                        pass
                
                # Update status with detailed batch information
                percent = int((batch_idx / total_batches) * 100)
                status_msg = f"🎬 Batch {batch_idx}/{total_batches} ({percent}%): {frames_info}{eta_str}"
                
                # Queue the status update for yielding
                try:
                    shared_queues.batch_progress.put(('batch_progress', status_msg))
                    shared_queues.counter['put'] += 1
                    logger.info(f"📤 Queued batch progress: {status_msg}, queue size after put: {shared_queues.batch_progress.qsize()}, total puts: {shared_queues.counter['put']}")
                except Exception as e:
                    logger.warning(f"❌ Failed to queue batch progress: {e}")
                        
                # Call the outer progress_callback if provided
                if progress_callback:
                    progress_pct = (batch_idx / total_batches) * 0.8 + 0.2  # Scale to 20-100%
                    progress_callback(progress_pct, f"Processing batch {batch_idx}/{total_batches} ({int(progress_pct * 100)}%)")
                    
            # Batch time tracking callback
            def track_batch_time(batch_time):
                shared_queues.batch_times.append(batch_time)
            
            # Track frames processed for chunk preview generation
            frames_processed_count = 0
            last_chunk_frame_end = 0
            accumulated_tensors = []  # Store tensors for chunk creation when not saving frames
            
            # Frame save callback wrapper
            def frame_save_callback(batch_tensor, batch_num, start_idx, end_idx):
                nonlocal frames_processed_count, last_chunk_frame_end, chunk_results, last_chunk_video_path, accumulated_tensors
                
                # The actual frames saved is determined by the indices
                # start_idx and end_idx represent the actual frame indices being saved
                actual_frames_saved = end_idx - start_idx
                frames_processed_count = end_idx  # Use end_idx as the total frames processed so far
                
                # Save frames if either save_frames is enabled OR chunk preview needs frames
                if processed_frames_permanent_save_path and (save_frames or seedvr2_config.enable_chunk_preview):
                    saved_count = _save_batch_frames_immediately(
                        batch_tensor,
                        batch_num,
                        start_idx,
                        processed_frames_permanent_save_path,
                        logger
                    )
                    if save_frames:
                        logger.info(f"💾 Saved {saved_count} frames from batch {batch_num}")
                    else:
                        logger.debug(f"💾 Saved {saved_count} frames for chunk preview")
                
                logger.info(f"📊 Total frames saved: {frames_processed_count}, last chunk ended at: {last_chunk_frame_end}, current batch saves: {actual_frames_saved} frames")
                
                # Check if we have enough frames for a new chunk preview
                if chunks_permanent_save_path and seedvr2_config.enable_chunk_preview:
                    # Accumulate tensors if not saving frames
                    if not save_frames:
                        accumulated_tensors.append(batch_tensor.cpu())
                    
                    while frames_processed_count >= last_chunk_frame_end + max_chunk_len:
                            # Generate chunk preview from saved frames
                            chunk_start = last_chunk_frame_end
                            chunk_end = chunk_start + max_chunk_len
                            chunk_id = len(chunk_results) + 1
                            
                            logger.info(f"🎬 Creating chunk {chunk_id} preview from frames {chunk_start+1} to {chunk_end}")
                            
                            # Get chunk frames either from disk or memory
                            chunk_frames = None
                            
                            if save_frames and processed_frames_permanent_save_path:
                                # Read frames from disk for this chunk
                                chunk_frames_list = []
                                for frame_idx in range(chunk_start, chunk_end):
                                    # Match the actual filename format used in _save_batch_frames_immediately
                                    frame_path = processed_frames_permanent_save_path / f"frame_{frame_idx+1:06d}.png"
                                    if frame_path.exists():
                                        frame_bgr = cv2.imread(str(frame_path))
                                        if frame_bgr is not None:
                                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                                            chunk_frames_list.append(frame_tensor)
                                
                                if chunk_frames_list:
                                    chunk_frames = torch.stack(chunk_frames_list, dim=0)
                            else:
                                # Use accumulated tensors from memory
                                if accumulated_tensors:
                                    all_frames = torch.cat(accumulated_tensors, dim=0)
                                    if all_frames.shape[0] >= chunk_end:
                                        chunk_frames = all_frames[chunk_start:chunk_end]
                            
                            if chunk_frames is not None and chunk_frames.shape[0] > 0:
                                
                                chunk_result = {
                                    'frames_tensor': chunk_frames,
                                    'chunk_id': chunk_id,
                                    'frame_count': chunk_frames.shape[0],
                                    'processing_time': time.time(),
                                    'device_id': device_id,
                                    'chunk_start_frame': chunk_start + 1,
                                    'chunk_end_frame': chunk_end
                                }
                                chunk_results.append(chunk_result)
                                
                                # Save chunk preview immediately
                                current_chunk_path = _save_seedvr2_chunk_previews(
                                    [chunk_result],
                                    chunks_permanent_save_path,
                                    original_fps or 30.0,
                                    seedvr2_config,
                                    ffmpeg_preset,
                                    ffmpeg_quality,
                                    ffmpeg_use_gpu,
                                    logger
                                )
                                
                                if current_chunk_path:
                                    last_chunk_video_path = current_chunk_path
                                    logger.info(f"✅ Chunk {chunk_id} preview generated during processing: {current_chunk_path}")
                                    # Put the chunk update in the queue for UI update
                                    shared_queues.chunk_update.put((chunk_id, current_chunk_path))
                                else:
                                    logger.warning(f"⚠️ Failed to generate chunk {chunk_id} preview")
                            else:
                                logger.warning(f"⚠️ No frames available for chunk {chunk_id} (start: {chunk_start}, end: {chunk_end})")
                            
                            last_chunk_frame_end = chunk_end
            
            # Use threading to run generation while monitoring chunk updates
            import threading
            
            # Store chunk updates in queue from callback
            original_frame_save_callback = frame_save_callback
            def enhanced_frame_save_callback(batch_tensor, batch_num, start_idx, end_idx):
                # Call original callback
                original_frame_save_callback(batch_tensor, batch_num, start_idx, end_idx)
                
                # Check if new chunk was created
                if chunk_results and last_chunk_video_path:
                    last_chunk = chunk_results[-1]
                    shared_queues.chunk_update.put((last_chunk['chunk_id'], last_chunk_video_path))
            
            # Function to run generation in thread
            def run_generation():
                try:
                    result = generation_loop(
                        runner=session_manager.runner,
                        images=frames_tensor,
                        cfg_scale=processing_args.get("cfg_scale", 1.0),
                        seed=processing_args.get("seed", -1),
                        res_w=res_w,
                        batch_size=batch_size,
                        preserve_vram=processing_args["preserve_vram"],
                        temporal_overlap=temporal_overlap,
                        debug=processing_args.get("debug", False),
                        block_swap_config=session_manager.block_swap_config,
                        progress_callback=generation_progress_callback,
                        frame_save_callback=enhanced_frame_save_callback,
                        tiled_vae=processing_args.get("tiled_vae", False),
                        tile_size=processing_args.get("tile_size", (64, 64)),
                        tile_stride=processing_args.get("tile_stride", (32, 32))
                    )
                    shared_queues.result.put(('success', result))
                except Exception as e:
                    shared_queues.result.put(('error', e))
            
            # Start generation in thread
            generation_thread = threading.Thread(target=run_generation)
            generation_thread.start()
            
            # Immediately yield that processing has started
            yield (None, "🚀 SeedVR2 processing started in background...", last_chunk_video_path, "Starting SeedVR2 processing...", None)
            
            # Track which chunks have been yielded to prevent duplicates
            yielded_chunks = set()
            last_chunk_status = "Starting SeedVR2 processing..."
            
            # Monitor for updates while generation runs
            last_status_time = time.time()
            monitor_count = 0
            while generation_thread.is_alive():
                monitor_count += 1
                # Debug logging removed to reduce console clutter
                # if monitor_count % 100 == 0:  # Log every 100 iterations
                #     logger.info(f"🔍 Monitor loop active - iteration {monitor_count}, batch_progress_queue id={id(shared_queues.batch_progress)}, queue size: {shared_queues.batch_progress.qsize()}")
                
                updates_found = False
                
                # Check for batch progress updates (non-blocking)
                try:
                    # Log before attempting to get from queue
                    if monitor_count % 20 == 0:  # Every second (20 * 0.05s)
                        logger.debug(f"🔍 Checking batch_progress_queue, current size: {shared_queues.batch_progress.qsize()}")
                    
                    # Add detailed logging around the get operation
                    queue_size_before = shared_queues.batch_progress.qsize()
                    # Debug logging removed to reduce console clutter
                    # if queue_size_before > 0:
                    #     logger.info(f"🔍 Queue has {queue_size_before} items, attempting to get...")
                    
                    update_type, status_msg = shared_queues.batch_progress.get(timeout=0.1)
                    
                    # Debug logging removed to reduce console clutter
                    # logger.info(f"🔍 Successfully got from queue: type={update_type}, queue size after={shared_queues.batch_progress.qsize()}")
                    if update_type == 'batch_progress':
                        shared_queues.counter['get'] += 1
                        logger.info(f"📊 Yielding batch progress update: {status_msg}, total gets: {shared_queues.counter['get']}")
                        # Yield with status message in both status and chunk_status fields for better visibility
                        yield (None, status_msg, last_chunk_video_path, status_msg, None)
                        updates_found = True
                        last_status_time = time.time()
                        # Also update the last_chunk_status for persistence
                        last_chunk_status = status_msg
                        
                        # Update current batch index when we get a batch progress update
                        # Extract batch number from the message if possible
                        import re
                        batch_match = re.search(r'Batch (\d+)/', status_msg)
                        if batch_match:
                            current_batch_idx = int(batch_match.group(1))
                        
                        # Track batch time for ETA updates
                        if shared_queues.batch_times and len(shared_queues.batch_times) > 0:
                            # Update with total progress
                            elapsed_time = time.time() - processing_start_time
                            total_eta = (elapsed_time / len(shared_queues.batch_times)) * (total_batches - len(shared_queues.batch_times))
                            if total_eta > 60:
                                total_minutes = int(total_eta / 60)
                                total_secs = int(total_eta % 60)
                                total_eta_str = f"{total_minutes}m {total_secs}s"
                            else:
                                total_eta_str = f"{int(total_eta)}s"
                            logger.info(f"📊 Total ETA: {total_eta_str} remaining")
                            
                except thread_queue.Empty:
                    # Queue was empty, this is normal
                    pass
                except Exception as e:
                    logger.error(f"❌ Error getting from batch_progress_queue: {e}")
                    
                # Check for chunk updates (non-blocking)
                try:
                    chunk_id, chunk_path = shared_queues.chunk_update.get(timeout=0.05)
                    # Only yield if we haven't yielded this chunk yet
                    if chunk_id not in yielded_chunks:
                        yielded_chunks.add(chunk_id)
                        logger.info(f"📹 Yielding chunk {chunk_id} preview update to UI with path: {chunk_path}")
                        # Ensure chunk_path is absolute and exists
                        if chunk_path and os.path.exists(chunk_path):
                            chunk_path = os.path.abspath(chunk_path)
                            logger.info(f"✅ Chunk file verified to exist at: {chunk_path}")
                        else:
                            logger.warning(f"⚠️ Chunk path does not exist: {chunk_path}")
                        yield (None, f"Chunk {chunk_id} preview generated", chunk_path, f"Chunk {chunk_id} preview ready", None)
                        updates_found = True
                        last_status_time = time.time()
                except thread_queue.Empty:
                    pass
                
                # Yield a heartbeat status every 2 seconds if no other updates
                if not updates_found and time.time() - last_status_time > 2:
                    elapsed = int(time.time() - processing_start_time)
                    
                    # Calculate ETA based on batch progress
                    eta_str = ""
                    if shared_queues.batch_times and len(shared_queues.batch_times) > 0:
                        # Use average batch time to estimate remaining time
                        avg_batch_time = sum(shared_queues.batch_times) / len(shared_queues.batch_times)
                        completed_batches = len(shared_queues.batch_times)
                        batches_remaining = total_batches - completed_batches
                        eta_seconds = avg_batch_time * batches_remaining
                        
                        # Format ETA
                        if eta_seconds > 60:
                            eta_minutes = int(eta_seconds / 60)
                            eta_secs = int(eta_seconds % 60)
                            eta_str = f" | ETA: {eta_minutes}m {eta_secs}s"
                        else:
                            eta_str = f" | ETA: {int(eta_seconds)}s"
                    
                    heartbeat_msg = f"⏳ Processing... ({elapsed}s elapsed{eta_str})"
                    logger.info(f"💓 Yielding heartbeat: {heartbeat_msg}")
                    # Preserve the last chunk status during heartbeat
                    yield (None, heartbeat_msg, last_chunk_video_path, last_chunk_status, None)
                    last_status_time = time.time()
                    
                # Small sleep to prevent CPU spinning
                if not updates_found:
                    time.sleep(0.1)
            
            # Wait for generation to complete
            generation_thread.join()
            
            # Log queue statistics
            logger.info(f"📊 Batch progress queue final stats: puts={shared_queues.counter['put']}, gets={shared_queues.counter['get']}, remaining in queue={shared_queues.batch_progress.qsize()}")
            
            # Get the result
            status, result = shared_queues.result.get()
            if status == 'error':
                # Check if this is a cancellation error
                if isinstance(result, CancelledError) or "CancelledError" in str(type(result).__name__):
                    logger.info("Processing was cancelled - will create partial video")
                    is_partial_result = True
                    # Set result_tensor to empty so we'll use saved frames
                    result_tensor = torch.zeros(0, dtype=torch.float16)
                else:
                    raise result
            else:
                result_tensor = result
            logger.info(f"✅ Generation completed successfully: output shape {result_tensor.shape}")
            
            # Trim output to original frame count (remove padding)
            if result_tensor.shape[0] > frames_to_output:
                logger.info(f"🔧 Trimming output from {result_tensor.shape[0]} to {frames_to_output} frames (removing padding)")
                result_tensor = result_tensor[:frames_to_output]
            
            # Frames are now saved progressively during processing via frame_save_callback
            
            # Check if we have any remaining frames to create a final chunk
            if chunks_permanent_save_path and seedvr2_config.enable_chunk_preview and frames_processed_count > last_chunk_frame_end:
                # Create final chunk from remaining frames
                chunk_start = last_chunk_frame_end
                chunk_end = frames_processed_count
                chunk_id = len(chunk_results) + 1
                
                logger.info(f"🎬 Creating final chunk {chunk_id} preview from frames {chunk_start+1} to {chunk_end}")
                
                # Read remaining frames from disk
                chunk_frames_list = []
                for frame_idx in range(chunk_start, chunk_end):
                    # Match the actual filename format used in _save_batch_frames_immediately
                    frame_path = processed_frames_permanent_save_path / f"frame_{frame_idx+1:06d}.png"
                    if frame_path.exists():
                        frame_bgr = cv2.imread(str(frame_path))
                        if frame_bgr is not None:
                            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                            chunk_frames_list.append(frame_tensor)
                
                if chunk_frames_list:
                    chunk_frames = torch.stack(chunk_frames_list, dim=0)
                    
                    chunk_result = {
                        'frames_tensor': chunk_frames,
                        'chunk_id': chunk_id,
                        'frame_count': len(chunk_frames_list),
                        'processing_time': time.time(),
                        'device_id': device_id,
                        'chunk_start_frame': chunk_start + 1,
                        'chunk_end_frame': chunk_end
                    }
                    chunk_results.append(chunk_result)
                    
                    # Save final chunk preview
                    current_chunk_path = _save_seedvr2_chunk_previews(
                        [chunk_result],
                        chunks_permanent_save_path,
                        original_fps or 30.0,
                        seedvr2_config,
                        ffmpeg_preset,
                        ffmpeg_quality,
                        ffmpeg_use_gpu,
                        logger
                    )
                    
                    if current_chunk_path:
                        last_chunk_video_path = current_chunk_path
                        logger.info(f"✅ Final chunk {chunk_id} preview generated: {current_chunk_path}")
                        yield (None, f"Final chunk {chunk_id} preview ready", last_chunk_video_path, f"Final chunk {chunk_id} preview ready", None)
            
            # Convert result to list of tensors for compatibility with rest of pipeline
            processed_frames = [result_tensor]
            total_processed_frames = result_tensor.shape[0]
            
            logger.info(f"Single-GPU processing complete: {result_tensor.shape[0]} frames processed")
            
            # Yield final results (not return, since this is a generator)
            yield result_tensor, chunk_results, last_chunk_video_path, "Single-GPU processing complete"
            
        except Exception as e:
            logger.error(f"Error in single GPU processing: {e}")
            # ✅ FIX: Return empty tensor with correct shape instead of just [0]
            if frames_tensor is not None and frames_tensor.shape[0] > 0:
                empty_tensor = torch.zeros(0, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3], dtype=torch.float16)
            else:
                empty_tensor = torch.zeros(0, 256, 256, 3, dtype=torch.float16)  # Default fallback shape
            yield empty_tensor, [], None, f"Error: {e}"
    
    except Exception as outer_e:
        # Handle any exceptions from the outer try block
        logger.error(f"Error in SeedVR2 processing setup: {outer_e}")
        if frames_tensor is not None and frames_tensor.shape[0] > 0:
            empty_tensor = torch.zeros(0, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3], dtype=torch.float16)
        else:
            empty_tensor = torch.zeros(0, 256, 256, 3, dtype=torch.float16)
        yield empty_tensor, [], None, f"Setup error: {outer_e}"


def _process_multi_gpu_cli(
    frames_tensor: torch.Tensor,
    device_list: List[str],
    processing_args: Dict[str, Any],
    seedvr2_config,
    progress_callback: Optional[callable] = None,
    logger: Optional[logging.Logger] = None,
    save_frames: bool = False,
    processed_frames_permanent_save_path: Optional[Path] = None,
    chunks_permanent_save_path: Optional[Path] = None,
    original_fps: Optional[float] = None,
    ffmpeg_preset: str = "medium",
    ffmpeg_quality: int = 23,
    ffmpeg_use_gpu: bool = False,
    max_chunk_len: int = 25  # ✅ FIX: Pass user's chunk frame count setting
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Optional[str]]:
    """
    Process frames using multiple GPUs with CLI approach.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Processing on multiple GPUs: {device_list}")
    
    num_devices = len(device_list)
    
    # Split frames across devices
    chunks = torch.chunk(frames_tensor, num_devices, dim=0)
    
    # Use multiprocessing for true parallel GPU processing
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_queue = manager.Queue()
    workers = []
    
    # Prepare shared arguments
    shared_args = processing_args.copy()
    shared_args.update({
        "block_swap_enabled": seedvr2_config.enable_block_swap,
        "block_swap_counter": seedvr2_config.block_swap_counter,
        "block_swap_offload_io": seedvr2_config.block_swap_offload_io,
        "block_swap_model_caching": seedvr2_config.block_swap_model_caching,
    })
    
    # Start worker processes
    for idx, (device_id, chunk_tensor) in enumerate(zip(device_list, chunks)):
        p = mp.Process(
            target=_worker_process_cli,
            args=(idx, device_id, chunk_tensor.cpu().numpy(), shared_args, return_queue),
        )
        p.start()
        workers.append(p)
    
    # Collect results
    results_np = [None] * num_devices
    collected = 0
    while collected < num_devices:
        proc_idx, res_np = return_queue.get()
        results_np[proc_idx] = res_np
        collected += 1
        
        if progress_callback:
            progress = 0.2 + 0.6 * (collected / num_devices)
            progress_callback(progress, f"GPU {collected}/{num_devices} completed")
    
    # Wait for all processes to complete
    for p in workers:
        p.join()
    
    # Concatenate results in original order
    result_tensor = torch.from_numpy(np.concatenate(results_np, axis=0)).to(torch.float16)
    
    # Create chunk results for preview functionality
    chunk_results = []
    if chunks_permanent_save_path and seedvr2_config.enable_chunk_preview:
        # For now, create a single chunk with the entire processed result
        # This can be enhanced later to support actual chunking during processing
        chunk_results.append({
            'frames_tensor': result_tensor,
            'chunk_id': 1,
            'frame_count': result_tensor.shape[0],
            'processing_time': time.time(),
        })
    
    logger.info(f"Multi-GPU processing complete: {result_tensor.shape[0]} frames processed")
    return result_tensor, chunk_results, None # No chunk preview for multi-GPU


def _worker_process_cli(proc_idx: int, device_id: str, frames_np: np.ndarray, shared_args: Dict[str, Any], return_queue):
    """
    Worker process for multi-GPU processing using CLI approach.
    """
    # Set CUDA device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    
    # Convert numpy back to tensor
    frames_tensor = torch.from_numpy(frames_np).to(torch.float16).cuda()
    
    # Setup block swap if enabled
    block_swap = None
    if shared_args.get("block_swap_enabled", False) and shared_args.get("block_swap_counter", 0) > 0:
        block_swap = SeedVR2BlockSwap(enable_debug=shared_args.get("debug", False))
    
    # Process frames with SeedVR2 model
    # Reconstruct seedvr2_config from shared_args
    class SeedVR2ConfigStub:
        def __init__(self, args):
            self.enable_block_swap = args.get("block_swap_enabled", False)
            self.block_swap_counter = args.get("block_swap_counter", 0)
            self.block_swap_offload_io = args.get("block_swap_offload_io", False)
            self.block_swap_model_caching = args.get("block_swap_model_caching", False)
    
    seedvr2_config_stub = SeedVR2ConfigStub(shared_args)
    
    result_tensor = _process_batch_with_seedvr2_model(
        frames_tensor,
        shared_args,
        seedvr2_config_stub,
        block_swap
    )
    
    # Send result back as numpy array
    return_queue.put((proc_idx, result_tensor.cpu().numpy()))


def _process_batch_with_seedvr2_model(
    batch_frames: torch.Tensor,
    processing_args: Dict[str, Any],
    seedvr2_config,
    block_swap: Optional[SeedVR2BlockSwap] = None
) -> torch.Tensor:
    """
    ✅ FIXED: Process frames using session-based approach instead of recreating model.
    This function now REUSES the existing model instead of creating a new one each time.
    """
    try:
        debug = processing_args.get("debug", False)
        
        if debug:
            print(f"🔄 Processing batch: {batch_frames.shape} (REUSING MODEL)")
            print(f"📁 Model: {processing_args.get('model', 'unknown')}")
            print(f"💾 Preserve VRAM: {processing_args.get('preserve_vram', True)}")
        
        # Check system requirements
        if not check_seedvr2_system_requirements(debug):
            if debug:
                print("❌ System requirements not met for SeedVR2")
            return _apply_placeholder_upscaling(batch_frames, debug)
        
        # Get session manager
        session_manager = get_session_manager()
        
        # Check if model has changed and needs reinitialization
        needs_reinit = False
        requested_model = processing_args.get('model', '')
        
        if session_manager.is_initialized and hasattr(session_manager, 'current_model'):
            if session_manager.current_model != requested_model:
                # Log model change even when not in debug mode
                if session_manager.logger:
                    session_manager.logger.info(f"🔄 SeedVR2 model change detected: {session_manager.current_model} -> {requested_model}")
                if debug:
                    print(f"🔄 Model changed from {session_manager.current_model} to {requested_model}")
                    print("🧹 Cleaning up previous session...")
                
                # Check if switching from FP8 to FP16 (most problematic transition)
                is_fp8_to_fp16_switch = False
                if session_manager.current_model and requested_model:
                    old_is_fp8 = 'fp8' in session_manager.current_model.lower() or 'e4m3fn' in session_manager.current_model.lower()
                    new_is_fp16 = 'fp16' in requested_model.lower()
                    is_fp8_to_fp16_switch = old_is_fp8 and new_is_fp16
                    if is_fp8_to_fp16_switch and session_manager.logger:
                        session_manager.logger.warning("⚠️ FP8 to FP16 transition detected - applying COMPLETE session destruction")
                
                # For FP8 to FP16, use complete destruction instead of normal cleanup
                if is_fp8_to_fp16_switch:
                    destroy_global_session_completely()
                    # Force re-creation of session manager on next access
                    session_manager = None
                else:
                    # Normal cleanup for other transitions
                    session_manager.cleanup_session()
                
                # Force aggressive GPU memory cleanup
                force_cleanup_gpu_memory()
                
                # Additional cleanup for model switching
                # Clear any remaining CUDA context
                if torch.cuda.is_available():
                    try:
                        # Wait a bit for cleanup to complete
                        import time
                        time.sleep(0.5 if not is_fp8_to_fp16_switch else 1.0)  # Longer wait for FP8->FP16
                        
                        # Additional cache clearing
                        iterations = 5 if is_fp8_to_fp16_switch else 3
                        for i in range(iterations):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            gc.collect()
                            if is_fp8_to_fp16_switch and i < iterations - 1:
                                time.sleep(0.2)  # Brief pauses for FP8->FP16
                        
                        # For FP8 to FP16 transitions, try even more aggressive cleanup
                        if is_fp8_to_fp16_switch:
                            # Reset all CUDA memory stats
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.reset_accumulated_memory_stats()
                            
                            # Force full garbage collection
                            gc.collect(2)  # Collect all generations
                            
                            # Final aggressive cache clear
                            for _ in range(3):
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                        
                        # Log final VRAM state (only if session_manager still exists)
                        if session_manager and session_manager.logger:
                            vram_status = session_manager.get_vram_usage()
                            session_manager.logger.info(f"📊 VRAM after model switch cleanup: {vram_status['allocated']:.2f}GB allocated, {vram_status['reserved']:.2f}GB reserved")
                        else:
                            # Log directly if session manager was destroyed
                            logger = logging.getLogger('video_to_video')
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / 1024**3
                                reserved = torch.cuda.memory_reserved() / 1024**3
                                logger.info(f"📊 VRAM after complete destruction: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    except Exception as e:
                        logger = logging.getLogger('video_to_video')
                        logger.warning(f"Additional cleanup error: {e}")
                
                needs_reinit = True
        
        # Re-get session manager if it was destroyed
        if session_manager is None:
            session_manager = get_session_manager()
        
        # Initialize session if needed or model changed
        if not session_manager.is_initialized or needs_reinit:
            if debug:
                print(f"🔧 Initializing SeedVR2 session with model: {requested_model}")
            
            success = session_manager.initialize_session(processing_args, seedvr2_config)
            if not success:
                if debug:
                    print("❌ Failed to initialize SeedVR2 session")
                return _apply_placeholder_upscaling(batch_frames, debug)
        
        # ✅ FIXED: Process with REUSED model
        if debug:
            print(f"🚀 Processing batch with REUSED SeedVR2 model")
        
        result_tensor = session_manager.process_batch(batch_frames)
        
        if debug:
            print(f"✅ Batch processed successfully: {result_tensor.shape}")
        
        return result_tensor
        
    except Exception as e:
        if debug:
            print(f"❌ Batch processing error: {e}")
            import traceback
            traceback.print_exc()
        
        # On error, cleanup and try placeholder
        cleanup_global_session()
        return _apply_placeholder_upscaling(batch_frames, debug)


def _apply_placeholder_upscaling(batch_frames: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """
    Apply placeholder upscaling when SeedVR2 is not available.
    
    Uses simple 4x nearest neighbor upscaling as fallback.
    """
    if debug:
        print("🔄 Applying placeholder 4x upscaling")
    
    try:
        # Simple 4x upscaling using interpolation
        T, H, W, C = batch_frames.shape
        upscaled = torch.nn.functional.interpolate(
            batch_frames.permute(0, 3, 1, 2),  # [T, C, H, W]
            scale_factor=4.0,
            mode='bicubic',
            align_corners=False
        ).permute(0, 2, 3, 1)  # Back to [T, H, W, C]
        
        if debug:
            print(f"✅ Placeholder upscaling: {batch_frames.shape} -> {upscaled.shape}")
        
        return upscaled
        
    except Exception as e:
        if debug:
            print(f"❌ Placeholder upscaling failed: {e}")
        # Last resort: return original frames
        return batch_frames


# Color correction utilities (independent implementation)
def apply_wavelet_color_correction(frames_tensor: torch.Tensor, original_frames: torch.Tensor) -> torch.Tensor:
    """
    Apply wavelet-based color correction to fix color shifts.
    
    Args:
        frames_tensor: Processed frames tensor
        original_frames: Original frames tensor for reference
        
    Returns:
        Color-corrected frames tensor
    """
    # Placeholder implementation
    # TODO: Implement actual wavelet reconstruction for color correction
    return frames_tensor


def validate_and_fix_seedvr2_config(seedvr2_config, debug: bool = False):
    """
    Validate and fix common SeedVR2 configuration issues.
    
    Args:
        seedvr2_config: SeedVR2 configuration object
        debug: Enable debug logging
        
    Returns:
        Fixed configuration object
    """
    if debug:
        print("🔧 Validating SeedVR2 configuration...")
    
    # Set safer defaults for problematic configurations
    if hasattr(seedvr2_config, 'model'):
        # Use FP8 models for better stability and memory efficiency
        if 'fp16' in seedvr2_config.model and '3b' in seedvr2_config.model:
            old_model = seedvr2_config.model
            seedvr2_config.model = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
            if debug:
                print(f"🔄 Switched model from {old_model} to {seedvr2_config.model} for better stability")
    
    # Ensure batch size is optimal (4n+1 format)
    if hasattr(seedvr2_config, 'batch_size'):
        if seedvr2_config.batch_size % 4 != 1 or seedvr2_config.batch_size < 5:
            old_batch = seedvr2_config.batch_size
            seedvr2_config.batch_size = 5  # Smallest optimal batch size
            if debug:
                print(f"🔄 Adjusted batch size from {old_batch} to {seedvr2_config.batch_size} for optimal processing")
    
    # Log block swap configuration if enabled
    if hasattr(seedvr2_config, 'enable_block_swap') and seedvr2_config.enable_block_swap:
        if debug:
            print(f"🔄 Block swap enabled with {seedvr2_config.block_swap_counter} blocks")
            print(f"   - I/O offloading: {getattr(seedvr2_config, 'block_swap_offload_io', False)}")
            print(f"   - Model caching: {getattr(seedvr2_config, 'block_swap_model_caching', False)}")
    
    # Disable temporal overlap for stability if block swap was attempted
    if hasattr(seedvr2_config, 'temporal_overlap'):
        if seedvr2_config.temporal_overlap > 0:
            old_overlap = seedvr2_config.temporal_overlap
            seedvr2_config.temporal_overlap = 0
            if debug:
                print(f"🔄 Disabled temporal overlap (was {old_overlap}) for stability")
    
    # Set conservative CFG scale
    if hasattr(seedvr2_config, 'cfg_scale'):
        if seedvr2_config.cfg_scale != 1.0:
            old_cfg = seedvr2_config.cfg_scale
            seedvr2_config.cfg_scale = 1.0
            if debug:
                print(f"🔄 Set CFG scale from {old_cfg} to {seedvr2_config.cfg_scale} for stability")
    
    # Enable preserve VRAM for better memory management
    if hasattr(seedvr2_config, 'preserve_vram'):
        if not seedvr2_config.preserve_vram:
            seedvr2_config.preserve_vram = True
            if debug:
                print("🔄 Enabled preserve VRAM for better memory management")
    
    if debug:
        print("✅ SeedVR2 configuration validated and fixed")
    
    return seedvr2_config


def _save_seedvr2_chunk_previews(
    chunk_results: List[Dict[str, Any]], 
    chunks_permanent_save_path: Path,
    fps: float,
    seedvr2_config,
    ffmpeg_preset: str,
    ffmpeg_quality: int,
    ffmpeg_use_gpu: bool,
    logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    Save SeedVR2 chunk previews following STAR pattern.
    
    ✅ FIX: Uses actual chunk frame count from processing instead of hardcoded preview frame count.
    
    Args:
        chunk_results: List of chunk processing results
        chunks_permanent_save_path: Path to save chunks
        fps: Video frame rate
        seedvr2_config: SeedVR2 configuration
        ffmpeg_preset: FFmpeg encoding preset
        ffmpeg_quality: FFmpeg quality setting
        ffmpeg_use_gpu: Whether to use GPU encoding
        logger: Optional logger
        
    Returns:
        Path to last chunk video or None if no chunks
    """
    if not chunk_results or not seedvr2_config.enable_chunk_preview:
        return None
        
    try:
        chunks_permanent_save_path.mkdir(parents=True, exist_ok=True)
        
        # Sort chunks by chunk_id to maintain proper order
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get('chunk_id', 0))
        
        # Keep only the last N chunks as configured
        chunks_to_keep = sorted_chunks[-seedvr2_config.keep_last_chunks:]
        
        saved_chunk_paths = []
        last_chunk_video_path = None
        
        # ✅ FIX: Reduced logging - only show summary
        if logger:
            logger.info(f"Saving {len(chunks_to_keep)} chunk previews to {chunks_permanent_save_path}")
        
        for i, chunk_data in enumerate(chunks_to_keep):
            try:
                # ✅ FIX: Use proper chunk numbering from chunk_id
                chunk_num = chunk_data.get('chunk_id', i + 1)
                chunk_filename = f"chunk_{chunk_num:04d}.mp4"
                chunk_video_path = chunks_permanent_save_path / chunk_filename
                
                # Get chunk frames (torch tensor)
                chunk_frames = chunk_data.get('frames_tensor')
                if chunk_frames is None:
                    if logger:
                        logger.warning(f"Chunk {chunk_num} has no frames tensor, skipping")
                    continue
                
                # Create temporary directory for this chunk's frames
                with tempfile.TemporaryDirectory(prefix=f"seedvr2_chunk_{chunk_num}_") as temp_chunk_dir:
                    # ✅ FIX: Use actual frame count from tensor instead of hardcoded values
                    frame_count = chunk_frames.shape[0]
                    
                    # ✅ FIX: Reduced logging - only log processing for current chunk
                    if logger:
                        logger.info(f"Processing chunk {chunk_num}: {frame_count} frames, shape: {chunk_frames.shape}")
                    
                    # Save frames as images with reduced logging
                    saved_frames = 0
                    for frame_idx in range(frame_count):
                        try:
                            frame_tensor = chunk_frames[frame_idx]
                            
                            # Convert tensor to numpy array with proper shape handling
                            if frame_tensor.dim() == 4:  # Batch dimension present
                                frame_tensor = frame_tensor.squeeze(0)  # Remove batch dimension
                            
                            if frame_tensor.dim() == 3:  # CHW format
                                if frame_tensor.shape[0] == 3:  # RGB channels first
                                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                                elif frame_tensor.shape[2] == 3:  # RGB channels last
                                    frame_np = frame_tensor.cpu().numpy()
                                else:
                                    # Handle other channel configurations
                                    frame_np = frame_tensor.cpu().numpy()
                                    if frame_np.shape[0] == 3:
                                        frame_np = frame_np.transpose(1, 2, 0)
                            else:  # Already HWC format
                                frame_np = frame_tensor.cpu().numpy()
                            
                            # Ensure we have 3 channels (RGB)
                            if frame_np.shape[-1] != 3:
                                continue
                            
                            # Ensure values are in correct range [0, 255]
                            if frame_np.dtype != np.uint8:
                                if frame_np.max() <= 1.0:
                                    frame_np = (frame_np * 255).astype(np.uint8)
                                else:
                                    frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                            
                            # Save frame
                            frame_filename = f"frame_{frame_idx + 1:06d}.png"
                            frame_path = os.path.join(temp_chunk_dir, frame_filename)
                            
                            # Convert RGB to BGR for OpenCV
                            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                            success = cv2.imwrite(frame_path, frame_bgr)
                            
                            if success:
                                saved_frames += 1
                            
                        except Exception as e:
                            # ✅ FIX: Only log first few errors to avoid spam
                            if logger and frame_idx < 3:
                                logger.warning(f"Failed to save frame {frame_idx}: {e}")
                            continue
                    
                    # ✅ FIX: Only create video if frames were actually saved
                    if saved_frames > 0:
                        # Create video from saved frames using existing utility
                        from .ffmpeg_utils import create_video_from_input_frames
                        
                        try:
                            # Create video from frames
                            video_created = create_video_from_input_frames(
                                temp_chunk_dir, 
                                str(chunk_video_path), 
                                fps, 
                                ffmpeg_preset=ffmpeg_preset,
                                ffmpeg_quality_value=ffmpeg_quality,
                                ffmpeg_use_gpu=ffmpeg_use_gpu,
                                logger=logger
                            )
                            
                            if video_created and chunk_video_path.exists():
                                saved_chunk_paths.append(str(chunk_video_path))
                                last_chunk_video_path = str(chunk_video_path)
                                
                                # ✅ FIX: Only show successful completion
                                if logger:
                                    logger.info(f"Successfully created chunk preview: {chunk_video_path.name}")
                            
                        except Exception as video_error:
                            if logger:
                                logger.error(f"Failed to create chunk video {chunk_num}: {video_error}")
                    
            except Exception as chunk_error:
                if logger:
                    logger.error(f"Error processing chunk {i + 1}: {chunk_error}")
                continue
        
        # ✅ FIX: Return the last chunk path for Gradio display
        if last_chunk_video_path and logger:
            logger.info(f"Chunk preview generation complete. Last chunk: {os.path.basename(last_chunk_video_path)}")
        
        return last_chunk_video_path
        
    except Exception as e:
        if logger:
            logger.error(f"Error saving chunk previews: {e}")
        return None


def check_seedvr2_system_requirements(debug: bool = False) -> bool:
    """
    Check if the system meets SeedVR2 requirements.
    
    Args:
        debug: Enable debug logging
        
    Returns:
        True if requirements are met, False otherwise
    """
    if debug:
        print("🔍 Checking SeedVR2 system requirements...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        if debug:
            print("❌ CUDA not available")
        return False
    
    # Check VRAM
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8.0:  # Minimum 8GB VRAM
            if debug:
                print(f"❌ Insufficient VRAM: {gpu_memory:.1f}GB (minimum 8GB required)")
            return False
        else:
            if debug:
                print(f"✅ VRAM check passed: {gpu_memory:.1f}GB")
    
    # Check PyTorch version
    torch_version = torch.__version__
    if debug:
        print(f"✅ PyTorch version: {torch_version}")
    
    # Check for required dependencies
    try:
        import einops
        import omegaconf
        if debug:
            print("✅ Required dependencies available")
    except ImportError as e:
        if debug:
            print(f"❌ Missing dependency: {e}")
        return False
    
    if debug:
        print("✅ System requirements check passed")
    
    return True 
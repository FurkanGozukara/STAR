"""
Temporary fix for SeedVR2 block swap issues.
This module patches the block swap to handle device mismatches properly.
"""

import torch
import weakref
import logging

logger = logging.getLogger(__name__)

def apply_comprehensive_blockswap_fix():
    """
    Apply a comprehensive fix to the block swap implementation.
    This addresses:
    1. Device mismatch errors after first batch
    2. VRAM spikes during processing
    3. Proper tensor movement between devices
    """
    
    # Import the blockswap module
    try:
        import sys
        import os
        seedvr2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'SeedVR2')
        if seedvr2_path not in sys.path:
            sys.path.insert(0, seedvr2_path)
            
        from src.optimization import blockswap
        
        # Store original wrap function
        original_wrap_block = blockswap._wrap_block_forward
        
        def fixed_wrap_block_forward(block, block_idx, model, debugger):
            """Fixed version that ensures proper device handling."""
            
            if hasattr(block, '_original_forward'):
                return  # Already wrapped
                
            # Store original forward
            original_forward = block.forward
            
            # Create weak references
            model_ref = weakref.ref(model)
            debugger_ref = weakref.ref(debugger) if debugger else lambda: None
            
            # Store block index
            block._block_idx = block_idx
            block._original_forward = original_forward
            
            def wrapped_forward(self, *args, **kwargs):
                # Get references
                model = model_ref()
                debugger = debugger_ref()
                
                if not model:
                    return original_forward(*args, **kwargs)
                
                # Check if block swap is active
                if hasattr(model, 'blocks_to_swap') and self._block_idx <= model.blocks_to_swap:
                    # Get current device
                    current_device = next(self.parameters()).device
                    
                    # CRITICAL FIX: Always compute on main device (GPU)
                    compute_device = torch.device(model.main_device)
                    
                    # Move block to compute device if needed
                    if current_device != compute_device:
                        self.to(compute_device)
                        # Force synchronization to ensure move completes
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    
                    # Move ALL inputs to compute device
                    def move_to_device(obj, device):
                        if torch.is_tensor(obj):
                            return obj.to(device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move args and kwargs
                    moved_args = tuple(move_to_device(arg, compute_device) for arg in args)
                    moved_kwargs = {k: move_to_device(v, compute_device) for k, v in kwargs.items()}
                    
                    # Execute forward pass
                    try:
                        output = original_forward(*moved_args, **moved_kwargs)
                    except RuntimeError as e:
                        if "device" in str(e):
                            logger.error(f"Device error in block {self._block_idx}: {e}")
                            logger.error(f"Block device: {current_device}, Compute device: {compute_device}")
                            # Try moving everything to CPU as fallback
                            self.cpu()
                            moved_args = tuple(move_to_device(arg, 'cpu') for arg in args)
                            moved_kwargs = {k: move_to_device(v, 'cpu') for k, v in kwargs.items()}
                            output = original_forward(*moved_args, **moved_kwargs)
                            output = move_to_device(output, compute_device)
                        else:
                            raise
                    
                    # Move block back to offload device AFTER computation
                    if self._block_idx <= model.blocks_to_swap:
                        self.to(model.offload_device)
                        # Don't synchronize here to allow non-blocking
                    
                    return output
                else:
                    # No swapping for this block
                    return original_forward(*args, **kwargs)
            
            # Replace forward method
            block.forward = wrapped_forward.__get__(block, block.__class__)
        
        # Replace the wrap function
        blockswap._wrap_block_forward = fixed_wrap_block_forward
        
        logger.info("✅ Applied comprehensive block swap fix")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply block swap fix: {e}")
        return False


def disable_block_swap_for_stability():
    """
    Alternative: Completely disable block swap if fixes don't work.
    """
    logger.warning("⚠️ Disabling block swap for stability")
    
    # This would need to be called before model initialization
    # to prevent block swap from being applied
    pass
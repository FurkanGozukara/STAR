# SeedVR2 Block Swap Fix Summary

## Issue
When using SeedVR2 with block swap enabled (e.g., block_swap=25), the entire model was being loaded into VRAM initially before block swapping began. This defeated the purpose of block swapping, which is designed to manage limited VRAM by keeping most blocks on CPU and only moving them to GPU as needed.

## Root Cause
In the `_configure_blocks` function in `blockswap.py`, blocks that were NOT designated for swapping (those with index > blocks_to_swap) were being immediately moved to GPU during initialization:

```python
if b > model.blocks_to_swap:
    block.to(device)  # This moved blocks to GPU immediately!
```

## Fix Implementation
The fix involves two key changes:

### 1. Keep ALL blocks on CPU initially
Modified `_configure_blocks` to keep all transformer blocks on the offload device (CPU) regardless of whether they will use block swapping:

```python
# Always move to offload device initially when block swap is active
block.to(offload_device, non_blocking=use_non_blocking)
```

### 2. Handle non-swapped blocks dynamically
Updated `_wrap_block_forward` to handle two types of blocks:
- **Blocks with index ≤ blocks_to_swap**: Use traditional block swapping (move to GPU → compute → move back to CPU)
- **Blocks with index > blocks_to_swap**: Move to GPU on first use and keep them there (one-time move)

## Benefits
1. **No VRAM spike**: Model no longer loads entirely into VRAM at initialization
2. **Progressive VRAM usage**: VRAM usage grows gradually as blocks are moved to GPU on-demand
3. **Better memory management**: Users with limited VRAM can now run larger models effectively
4. **Maintains performance**: Blocks that don't need swapping stay on GPU after first use

## How Block Swap Now Works
1. Model loads entirely to CPU
2. During inference:
   - Blocks 0 to `blocks_to_swap` use dynamic swapping (GPU ↔ CPU)
   - Blocks > `blocks_to_swap` move to GPU once when first needed and stay there
3. VRAM usage remains controlled throughout the process

## Testing Recommendations
Test with different block_swap values to verify:
- No initial VRAM spike when model loads
- Progressive VRAM usage during inference
- Correct functioning with various model sizes (3B/7B)
- Performance remains acceptable
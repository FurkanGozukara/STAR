# CogVLM Quantization Fix for Linux Systems

## Problem Description

On Linux systems, CogVLM2 model loading with 4-bit or 8-bit quantization was failing with the following error:

```
ValueError: `.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the model has already been set to the correct devices and casted to the correct `dtype`.
```

Additionally, attempting to fix this by passing a `device` parameter to the model caused a new error on both Windows and Linux:

```
TypeError: CogVLMVideoForCausalLM.__init__() got an unexpected keyword argument 'device'
```

## Root Cause

The issue was caused by using `device_map="auto"` parameter when loading quantized models. This parameter triggers the accelerate library to automatically dispatch the model to devices, which internally calls `.to()` methods that are not supported for quantized models.

The CogVLM model constructor also doesn't accept a `device` parameter directly, so alternative approaches that pass device parameters fail.

## Solution

### Changes Made to `logic/cogvlm_utils.py`

1. **Removed `device_map="auto"` for quantized models**:
   - Previously: Used `device_map="auto"` for quantized models (caused Linux .to() error)
   - Now: Use `device_map=None` for quantized models

2. **Let bitsandbytes handle device placement automatically**:
   - Don't pass `device` parameter (unsupported by CogVLM)
   - Don't use `device_map="auto"` (causes .to() errors)
   - Let bitsandbytes quantization config handle device placement internally

3. **Separated device placement logic**:
   - Quantized models: Device placement handled automatically by bitsandbytes
   - Non-quantized models: Manual device placement using `.to()` after loading

4. **Improved error handling and logging**:
   - Added detailed logging for model loading parameters
   - Better separation of quantized vs non-quantized model handling

### Key Code Changes

```python
# Before (causing errors):
if bnb_config and device == 'cuda':
    current_device_map = "auto"  # Caused .to() error on Linux

model_args['device'] = target_device  # Caused TypeError on both systems

# After (fixed):
if bnb_config:
    current_device_map = None  # Let bitsandbytes handle device placement
    if logger:
        logger.info(f"Quantized model: letting bitsandbytes handle device placement automatically")

# Only include supported parameters
model_args = {
    'pretrained_model_name_or_path': cog_vlm_model_path,
    'torch_dtype': model_dtype if device == 'cuda' else torch.float32,
    'trust_remote_code': True,
    'low_cpu_mem_usage': effective_low_cpu_mem_usage
}

if bnb_config:
    model_args['quantization_config'] = bnb_config

# Only add device_map if it's not None (avoid for quantized models)
if current_device_map is not None:
    model_args['device_map'] = current_device_map

model = AutoModelForCausalLM.from_pretrained(**model_args)
```

## Testing

A test script `test_cogvlm_quantization_fix.py` was created to validate the fix:

```bash
python test_cogvlm_quantization_fix.py
```

The test validates:
- 4-bit quantization loading
- 8-bit quantization loading  
- Non-quantized model loading
- Proper device placement
- Model unloading

## Compatibility

This fix maintains compatibility with:
- ✅ Windows systems (no unsupported parameters)
- ✅ Linux systems (no .to() calls on quantized models)
- ✅ Both CUDA and CPU devices
- ✅ All quantization modes (4-bit, 8-bit, no quantization)

## Dependencies

The fix works with the following library versions:
- `transformers` >= 4.20.0
- `bitsandbytes` >= 0.41.0
- `accelerate` >= 0.20.0
- `torch` >= 2.0.0

## Verification

After applying the fix, CogVLM2 model loading should work successfully on both Windows and Linux systems. The logs should show:

```
INFO - Quantized model: letting bitsandbytes handle device placement automatically
INFO - Quantized model loaded - device placement handled by bitsandbytes
INFO - CogVLM2 model loaded (Quant: 4, Requested Device: cuda, Final Device(s): cuda:0, Dtype: torch.uint8).
```

## Related Files

- `logic/cogvlm_utils.py` - Main fix implementation
- `test_cogvlm_quantization_fix.py` - Validation test script
- `secourses_app.py` - Application that uses CogVLM functionality

## Notes

- This fix only affects quantized model loading (4-bit/8-bit)
- Non-quantized models continue to use the previous loading method with manual device placement
- The fix is backwards compatible and doesn't break existing functionality
- Memory usage and performance characteristics remain unchanged
- The solution relies on bitsandbytes to handle device placement automatically for quantized models 
# Bug Analysis and Fixes for UnboundLocalError and Frame Processing Issues

## Issues Identified from Log Analysis

### 1. **Critical Error: `UnboundLocalError: local variable 'final_chunk_video_path' referenced before assignment`**

**Location:** `logic/scene_processing_core.py` line 974

**Root Cause:** 
- The variable `final_chunk_video_path` was only being assigned when:
  1. `save_chunks` is enabled AND
  2. `frames_for_this_video_chunk` has frames (meaning frames were successfully found and copied)
- However, the metadata section was trying to reference this variable regardless of whether it was assigned
- When no frames were found for chunk video creation, the variable remained undefined, causing the UnboundLocalError

**Log Evidence:**
```
19:01:49 - WARNING - No frames for scene chunk 1/4, video not created.
19:01:49 - ERROR - Error processing scene 1: local variable 'final_chunk_video_path' referenced before assignment
```

### 2. **Frame Missing Issue: "Src frame not found for scene chunk video"**

**Root Cause:** 
- The STAR diffusion and VAE decoding completed successfully
- However, when creating chunk videos, the expected frames were not found in the output directory
- This suggests a potential issue with frame naming, timing, or file I/O

**Log Evidence:**
```
19:01:45 - temporal vae decoding, finished in 00:12 total.
19:01:49 - WARNING - Src frame E:\Ultimate_Video_Processing_v1\STAR\tmp\star_run_1750867079_8203\scene_0001\output_frames\frame_000001.png not found for scene chunk video.
[...20 similar warnings for frames 1-20...]
19:01:49 - WARNING - No frames for scene chunk 1/4, video not created.
```

### 3. **CV2 Image Depth Warning**

**Warning:** `Unsupported depth image for selected encoder is fallbacked to CV_8U`

**Root Cause:** 
- The frame tensors may not be in the correct uint8 format when being saved with cv2.imwrite()
- CV2 automatically falls back to uint8, but this could indicate data type inconsistencies

## Fixes Implemented

### Fix 1: Initialize `final_chunk_video_path` with Default Value

**File:** `logic/scene_processing_core.py`

**Changes:**
- Added `final_chunk_video_path = None` initialization at the beginning of both chunk processing sections (context and regular)
- Added an `else` clause to initialize the variable when `save_chunks` is disabled
- This ensures the variable is always defined before being referenced in metadata

```python
# Before chunk processing
final_chunk_video_path = None  # Initialize with default value

# ...processing logic...

# At the end if save_chunks is disabled
else:
    # Initialize final_chunk_video_path for metadata even when save_chunks is disabled
    final_chunk_video_path = None
```

### Fix 2: Enhanced Frame Processing Debugging

**File:** `logic/scene_processing_core.py`

**Changes:**
- Added verification after each frame write to ensure frames are actually saved to disk
- Added logging to show how many frames were written and their location
- Added debugging info before chunk video creation to show expected vs actual frames

```python
# Verify frame was actually written
if not os.path.exists(frame_output_path):
    logger.error(f"Failed to write frame: {frame_output_path}")
else:
    logger.debug(f"Successfully wrote frame: {frame_output_path}")

# Debug: List what frames are actually in the output directory
if os.path.exists(scene_output_frames_dir):
    actual_frames = os.listdir(scene_output_frames_dir)
    logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: {len(actual_frames)} frames written to {scene_output_frames_dir}")
```

### Fix 3: Ensure Proper Image Format for CV2

**File:** `logic/scene_processing_core.py`

**Changes:**
- Added explicit dtype checking and conversion to uint8 before cv2.imwrite()
- This should prevent the CV2 depth warning and ensure consistent image format

```python
# Ensure the frame is uint8 format to avoid CV2 depth issues
if frame_np_hwc_uint8.dtype != np.uint8:
    frame_np_hwc_uint8 = frame_np_hwc_uint8.astype(np.uint8)
```

### Fix 4: Enhanced Chunk Video Creation Debugging

**File:** `logic/scene_processing_core.py`

**Changes:**
- Added logging to show how many frames are expected vs found during chunk video creation
- Added debug output of expected frame names to help diagnose naming mismatches

```python
logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: Looking for {len(output_frame_names)} frames in {scene_output_frames_dir}")
logger.debug(f"Expected frame names: {output_frame_names[:3]}...")  # Show first 3 expected names
logger.info(f"Scene {scene_index + 1} Chunk {chunk_idx+1}: Found {len(frames_for_this_video_chunk)} out of {len(output_frame_names)} expected frames")
```

## Impact of Fixes

### Immediate Impact:
1. **Eliminates UnboundLocalError:** The app will no longer crash with the `final_chunk_video_path` error
2. **Improved Debugging:** Enhanced logging will help identify the root cause of frame missing issues
3. **Better Image Format Handling:** Should reduce CV2 warnings and ensure consistent frame formats

### For Future Debugging:
1. **Frame Write Verification:** Will immediately detect if frames aren't being written to disk
2. **Frame Count Tracking:** Will show exactly how many frames are processed vs expected
3. **Directory Content Verification:** Will show what's actually in the output directory vs what's expected

## Recommended Next Steps

1. **Test the fixes** by running the same video that caused the original error
2. **Monitor the enhanced logging** to see if frames are being written correctly
3. **If frame missing issues persist**, investigate:
   - File system permissions
   - Disk space issues
   - Timing issues between frame writing and reading
   - Frame naming consistency

## Memory Usage Considerations

The user's logs show they have a 32GB RTX 5090, so the processing should have sufficient GPU memory. The error occurs after successful diffusion and VAE decoding, suggesting the issue is in the frame I/O rather than GPU memory constraints. 
# PHASE 1: ANALYSIS & UNDERSTANDING - SLIDING WINDOW REPLACEMENT

## Current Sliding Window Implementation Analysis

### **Current Logic (Traditional Sliding Window)**
- **Parameters:** `window_size` and `window_step`
- **Default Values:** `window_size=32`, `window_step=16` (50% overlap)
- **Behavior:** 
  - Process `window_size` frames at a time
  - Advance by `window_step` frames for next window
  - Creates overlapping processing windows
  - Complex logic for handling overlaps and frame extraction

### **New Logic Required (Context-Based Sliding Window)**
- **Parameters:** `max_chunk_len` (existing) and `context_overlap` (new)
- **Example:** `max_chunk_len=16`, `context_overlap=8`
- **Behavior:**
  - **Chunk 1:** Process frames 1-16, Output frames 1-16 (16 frames)
  - **Chunk 2:** Process frames 9-24, Output frames 17-24 (8 frames, 8 context)
  - **Chunk 3:** Process frames 17-32, Output frames 25-32 (8 frames, 8 context)
  - **Chunk 4:** Process frames 25-40, Output frames 33-40 (8 frames, 8 context)

---

## **FILES REQUIRING MODIFICATION**

### **1. secourses_app.py (Main App File)**
**Current UI Components:**
- `enable_sliding_window_check` (checkbox)
- `window_size_num` (slider: 2-256, default=32)
- `window_step_num` (slider: 1-128, default=16)

**New UI Components Needed:**
- `enable_context_window_check` (checkbox, rename existing)
- `context_overlap_num` (slider: 0 to max_chunk_len-1, default=8)

**Functions to Update:**
- `upscale_director_logic()` - parameter lists
- `process_batch_videos_wrapper()` - parameter lists
- All Gradio click handlers
- UI change handlers (`.change()` functions)

**Parameter Flow Points:**
- Lines 840, 988, 1129, 1253, 1308, 1345 - parameter passing
- Lines 793, 2370 - UI change handlers
- Lines 2416, 2479, 2528, 2583 - input/output parameter lists

---

### **2. logic/upscaling_core.py (Core Processing)**
**Current Implementation:**
- **Function:** `run_upscale()` (line 47)
- **Parameters:** `enable_sliding_window, window_size, window_step`
- **Logic Block:** Lines 736-1200 (complex sliding window processing)

**Required Changes:**
- **Function Signature:** Replace parameters with `enable_context_window, context_overlap`
- **Logic Replacement:** Replace entire sliding window block with context-based processing
- **Features to Maintain:**
  - Chunk saving (`save_chunks`)
  - RIFE interpolation for chunks
  - Progress reporting
  - Memory management
  - Frame saving compatibility

---

### **3. logic/scene_processing_core.py (Scene Processing)**
**Current Implementation:**
- **Function:** `process_single_scene()` (line 18)
- **Parameters:** `enable_sliding_window, window_size, window_step`
- **Logic Block:** Lines 291-650 (scene-specific sliding window)

**Required Changes:**
- **Function Signature:** Update parameters
- **Scene-Specific Logic:** Adapt context processing for scenes
- **Features to Maintain:**
  - Scene chunk saving
  - Scene progress callbacks
  - Auto-captioning per scene
  - Scene metadata recording

---

### **4. logic/batch_operations.py (Batch Processing)**
**Current Implementation:**
- **Function:** `process_batch_videos()` (line 70)
- **Parameters:** `enable_sliding_window_check_val, window_size_num_val, window_step_num_val`

**Required Changes:**
- **Parameter Update:** Replace with `enable_context_window_check_val, context_overlap_num_val`
- **Parameter Passing:** Update call to `run_upscale_func`

---

### **5. logic/config.py (Configuration)**
**Current Implementation:**
```python
DEFAULT_ENABLE_SLIDING_WINDOW = False
DEFAULT_WINDOW_SIZE = 32
DEFAULT_WINDOW_STEP = 16
```

**Required Changes:**
```python
DEFAULT_ENABLE_CONTEXT_WINDOW = False
DEFAULT_CONTEXT_OVERLAP = 8
```

**Exports Update:** Update `__all__` list to replace old constants

---

### **6. logic/metadata_handler.py (Metadata Recording)**
**Current Implementation:**
- Lines 26-28: Records sliding window parameters in metadata
- Conditional metadata based on `enable_sliding_window`

**Required Changes:**
- Replace sliding window metadata fields
- Update conditional logic for context window
- Maintain backward compatibility in metadata structure

---

### **7. logic/context_sliding_window.py (Helper Module)**
**Current Status:** Already exists with some helpful functions

**Required Enhancements:**
- `calculate_context_chunks()` - main chunk calculation function
- `process_context_chunk()` - individual chunk processing
- `optimize_context_parameters()` - parameter optimization
- Integration with existing helper functions

---

## **PARAMETER FLOW MAPPING**

### **Current Parameter Flow:**
```
secourses_app.py (UI) 
  ↓ enable_sliding_window_check_val, window_size_num_val, window_step_num_val
upscale_director_logic() 
  ↓ enable_sliding_window, window_size, window_step
logic/upscaling_core.py::run_upscale()
  ↓ enable_sliding_window, window_size, window_step
logic/scene_processing_core.py::process_single_scene()
```

### **New Parameter Flow:**
```
secourses_app.py (UI)
  ↓ enable_context_window_check_val, context_overlap_num_val
upscale_director_logic()
  ↓ enable_context_window, context_overlap
logic/upscaling_core.py::run_upscale()
  ↓ enable_context_window, context_overlap
logic/scene_processing_core.py::process_single_scene()
```

---

## **CURRENT SLIDING WINDOW LOGIC ANALYSIS**

### **Key Functions in upscaling_core.py:**
1. **`map_window_to_chunks()`** - Maps window ranges to chunk indices
2. **`get_chunk_frame_range()`** - Gets frame range for chunks
3. **`is_chunk_complete()`** - Checks if chunk is complete
4. **`get_effective_chunk_mappings()`** - Calculates chunk boundaries
5. **`save_effective_sliding_window_chunk()`** - Saves chunk videos

### **Processing Flow:**
1. Calculate window indices: `range(0, frame_count, window_step)`
2. Process each window with overlap handling
3. Extract frames from overlapped regions
4. Save chunks based on "effective chunk mappings"
5. Handle incomplete chunks at the end

### **Memory Management:**
- Tracks processed frames: `processed_frames_tracker`
- Saves chunks when complete: `saved_chunks` set
- Clears GPU memory after each window
- Immediate frame saving for performance

---

## **NEW CONTEXT LOGIC REQUIREMENTS**

### **Core Algorithm:**
```python
def calculate_context_chunks(total_frames, max_chunk_len, context_overlap):
    chunks = []
    current_output_start = 0
    chunk_index = 0
    
    while current_output_start < total_frames:
        if chunk_index == 0:
            # First chunk: no context, full output
            process_start = 0
            process_end = min(max_chunk_len, total_frames)
            output_start = 0
            output_end = process_end
        else:
            # Subsequent chunks: include context
            output_size = min(max_chunk_len - context_overlap, total_frames - current_output_start)
            if output_size <= 0:
                break
                
            output_start = current_output_start
            output_end = current_output_start + output_size
            process_start = max(0, output_start - context_overlap)
            process_end = output_end
        
        chunks.append({
            'chunk_idx': chunk_index,
            'process_start': process_start,
            'process_end': process_end, 
            'output_start': output_start,
            'output_end': output_end,
            'output_start_offset': output_start - process_start,
            'output_end_offset': output_end - process_start,
            'context_frames': context_overlap if chunk_index > 0 else 0
        })
        
        current_output_start = output_end
        chunk_index += 1
    
    return chunks
```

### **Processing Logic:**
```python
def process_context_chunk(chunk_info, all_frames, model, ...):
    # Get frames for processing (including context)
    process_frames = all_frames[chunk_info['process_start']:chunk_info['process_end']]
    
    # Process all frames through model
    processed_frames = model.process(process_frames)
    
    # Extract only output frames (exclude context)
    output_frames = processed_frames[chunk_info['output_start_offset']:chunk_info['output_end_offset']]
    
    return output_frames
```

---

## **COMPATIBILITY REQUIREMENTS**

### **Must Work With:**
1. **Scene Splitting** - Context windows within scenes
2. **Batch Processing** - Multiple videos with context
3. **RIFE Interpolation** - Apply to context-processed chunks  
4. **Tiling** - Spatial tiling + temporal context
5. **Frame Saving** - Save input/output frames correctly
6. **Chunk Saving** - Save chunk videos with context info
7. **Metadata Recording** - Record context parameters
8. **Progress Reporting** - Accurate progress with context overhead

### **Edge Cases to Handle:**
1. **Short Videos** - Fewer frames than `max_chunk_len`
2. **Context > Available** - Cap context to available frames  
3. **Last Chunk** - Handle remaining frames correctly
4. **Context = 0** - Disable context (same as current chunked mode)
5. **Context ≥ max_chunk_len** - Validation and error handling

---

## **PERFORMANCE CONSIDERATIONS**

### **Memory Impact:**
- **Context frames:** Additional frames loaded per chunk (except first)
- **Processing overhead:** More frames processed but fewer output
- **GPU memory:** Context frames use additional VRAM during processing

### **Speed Impact:**
- **Slower:** More total frames processed due to overlap
- **Better quality:** Improved temporal consistency
- **Optimization:** Efficient context frame management

### **VRAM Optimization:**
- Process context + output frames together
- Clear context frames immediately after processing
- Maintain existing VAE chunking for memory management

---

## **TESTING SCENARIOS**

### **Basic Test Cases:**
1. **Single video, context=8, max_chunk=16** - Basic functionality
2. **Batch videos with context** - Multiple video processing
3. **Scene split + context** - Combined features
4. **RIFE + context** - Interpolation with context chunks
5. **Very short video (10 frames)** - Edge case handling
6. **Context=0** - Disabled context mode
7. **Large context (15 with max_chunk=16)** - Near-maximum context

### **Expected Results:**
- Frame numbering must be exact and sequential
- No gaps or overlaps in output frames
- Chunk videos contain correct frame ranges
- Progress reporting matches actual processing
- Memory usage reasonable and stable

---

## **MIGRATION STRATEGY**

### **Backward Compatibility:**
- Keep old parameter names in metadata for comparison
- Provide migration path for existing users
- Clear documentation of changes

### **Default Values:**
- Set `context_overlap = max_chunk_len // 4` as reasonable default
- Provide UI hints for optimal values
- Auto-adjust if context ≥ max_chunk_len

---

## **SUCCESS CRITERIA FOR PHASE 1**

✅ **Complete understanding of current implementation**
✅ **All files and functions mapped**  
✅ **Parameter flow documented**
✅ **New logic algorithm defined**
✅ **Compatibility requirements identified**
✅ **Edge cases documented**
✅ **Testing scenarios planned**

**Ready for Phase 2: UI Changes** 
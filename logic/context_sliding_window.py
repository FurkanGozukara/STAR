"""
Context-Based Sliding Window Processing Module

This module implements a context-based sliding window approach similar to "Optimize Last Chunk Quality" 
but applied to all chunks. Each chunk (except first) includes previous frames as context but only 
outputs new frames.

Example with 16-frame chunks and 8-frame context:
- Chunk 1: Process frames 1-16, Output frames 1-16 (16 frames)
- Chunk 2: Process frames 9-24, Output frames 17-24 (8 frames, 8 context)  
- Chunk 3: Process frames 17-32, Output frames 25-32 (8 frames, 8 context)
- Chunk 4: Process frames 25-40, Output frames 33-40 (8 frames, 8 context)

Consistent context size: Always process (context_overlap + output_size) frames per chunk after the first.
"""

import os
import gc
import torch
import numpy as np
from typing import List, Generator, Tuple, Any, Optional, Dict
import math


def calculate_context_chunks(total_frames: int, max_chunk_len: int, context_overlap: int) -> List[Dict[str, int]]:
    """
    Calculate chunk boundaries for context-based sliding window processing.
    
    This implements the user's specification where:
    - First chunk: Process and output max_chunk_len frames (no context)
    - Subsequent chunks: Include context_overlap frames as context, output (max_chunk_len - context_overlap) new frames
    
    Example with 16-frame chunks and 8-frame context for 126 total frames:
    - Chunk 1: Process frames 1-16, Output frames 1-16 (16 frames)
    - Chunk 2: Process frames 9-24, Output frames 17-24 (8 new frames, 8 context)
    - Chunk 3: Process frames 17-32, Output frames 25-32 (8 new frames, 8 context)
    - Chunk 4: Process frames 25-40, Output frames 33-40 (8 new frames, 8 context)
    
    Args:
        total_frames: Total number of frames in the video (0-based indexing internally)
        max_chunk_len: Maximum frames per chunk (output size for first chunk)
        context_overlap: Number of previous frames to include as context
        
    Returns:
        List of chunk dictionaries with processing and output information
    """
    if total_frames <= 0 or max_chunk_len <= 0:
        return []
    
    # Ensure context_overlap doesn't exceed max_chunk_len
    effective_context_overlap = min(context_overlap, max_chunk_len)
    
    chunks = []
    current_output_start = 0
    chunk_index = 0
    
    while current_output_start < total_frames:
        if chunk_index == 0:
            # First chunk: process and output max_chunk_len frames
            output_start = current_output_start
            output_end = min(current_output_start + max_chunk_len, total_frames)
            process_start = output_start
            process_end = output_end
            
            # Next chunk starts where this one ends
            next_output_start = output_end
        else:
            # Subsequent chunks: output (max_chunk_len - context_overlap) new frames
            output_size = max_chunk_len - effective_context_overlap
            
            # Ensure we don't exceed remaining frames
            remaining_frames = total_frames - current_output_start
            if remaining_frames <= 0:
                break
                
            # Actual output size is minimum of desired output size and remaining frames
            actual_output_size = min(output_size, remaining_frames)
            
            output_start = current_output_start
            output_end = current_output_start + actual_output_size
            
            # Processing includes context + output frames
            process_start = max(0, output_start - effective_context_overlap)
            process_end = output_end
            
            # Next chunk starts where this one ends
            next_output_start = output_end
        
        # Calculate offsets within the processed chunk for extracting output frames
        output_start_offset = output_start - process_start
        output_end_offset = output_end - process_start
        
        chunk_info = {
            'chunk_idx': chunk_index,
            'process_start_idx': process_start,
            'process_end_idx': process_end,
            'process_length': process_end - process_start,
            'output_start_idx': output_start,
            'output_end_idx': output_end,
            'output_length': output_end - output_start,
            'output_start_offset': output_start_offset,
            'output_end_offset': output_end_offset,
            'has_context': chunk_index > 0,
            'context_frames': effective_context_overlap if chunk_index > 0 else 0
        }
        
        chunks.append(chunk_info)
        
        # Move to next chunk
        current_output_start = next_output_start
        chunk_index += 1
    
    return chunks


def validate_context_parameters(max_chunk_len: int, context_overlap: int, total_frames: int) -> Tuple[bool, str]:
    """
    Validate context-based sliding window parameters.
    
    Args:
        max_chunk_len: Maximum frames per chunk
        context_overlap: Number of context frames
        total_frames: Total frames in video
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_chunk_len <= 0:
        return False, "Max chunk length must be greater than 0"
    
    if context_overlap < 0:
        return False, "Context overlap cannot be negative"
    
    if total_frames <= 0:
        return False, "Total frames must be greater than 0"
    
    if context_overlap >= max_chunk_len:
        return False, f"Context overlap ({context_overlap}) must be less than max chunk length ({max_chunk_len})"
    
    # Check if we'll have enough frames for meaningful processing
    if total_frames < max_chunk_len and context_overlap > 0:
        return False, f"For videos with {total_frames} frames, context overlap should be 0 when max chunk length is {max_chunk_len}"
    
    return True, "Parameters are valid"


def log_context_chunks_info(chunks: List[Dict[str, int]], context_overlap: int, max_chunk_len: int, logger) -> None:
    """
    Log detailed information about context-based chunk processing.
    
    Args:
        chunks: List of chunk information dictionaries
        context_overlap: Context overlap setting
        max_chunk_len: Maximum chunk length setting
        logger: Logger instance
    """
    if not chunks:
        logger.warning("No chunks to process!")
        return
    
    total_chunks = len(chunks)
    total_frames = chunks[-1]['output_end_idx']
    
    logger.info(f"=== Context-Based Sliding Window Configuration ===")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Max chunk length (output): {max_chunk_len}")
    logger.info(f"Context overlap: {context_overlap}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info("")
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        process_range = f"{chunk['process_start_idx']+1}-{chunk['process_end_idx']}"
        output_range = f"{chunk['output_start_idx']+1}-{chunk['output_end_idx']}"
        
        if chunk['has_context']:
            context_info = f" (with {chunk['context_frames']} context frames)"
        else:
            context_info = " (no context)"
        
        logger.info(f"Chunk {chunk_num}/{total_chunks}: "
                   f"Process frames {process_range} ({chunk['process_length']} frames), "
                   f"Output frames {output_range} ({chunk['output_length']} frames)"
                   f"{context_info}")
    
    logger.info("=" * 50)


def extract_output_frames_from_context_chunk(processed_frames: List[Any], chunk_info: Dict[str, int]) -> List[Any]:
    """
    Extract only the output frames from a processed chunk that includes context.
    
    Args:
        processed_frames: List of processed frame tensors/arrays
        chunk_info: Chunk information dictionary
        
    Returns:
        List of frames that should be output (excluding context frames)
    """
    start_offset = chunk_info['output_start_offset']
    end_offset = chunk_info['output_end_offset']
    
    if start_offset >= len(processed_frames) or end_offset > len(processed_frames):
        raise ValueError(f"Invalid frame extraction: offset range {start_offset}-{end_offset} "
                        f"exceeds processed frames length {len(processed_frames)}")
    
    return processed_frames[start_offset:end_offset]


def get_context_processing_stats(chunks: List[Dict[str, int]]) -> Dict[str, Any]:
    """
    Get statistics about context-based processing.
    
    Args:
        chunks: List of chunk information dictionaries
        
    Returns:
        Dictionary with processing statistics
    """
    if not chunks:
        return {}
    
    total_chunks = len(chunks)
    total_output_frames = chunks[-1]['output_end_idx']
    total_processing_frames = sum(chunk['process_length'] for chunk in chunks)
    chunks_with_context = sum(1 for chunk in chunks if chunk['has_context'])
    total_context_frames = sum(chunk['context_frames'] for chunk in chunks)
    
    # Calculate efficiency metrics
    processing_overhead = total_processing_frames - total_output_frames
    efficiency_ratio = total_output_frames / total_processing_frames if total_processing_frames > 0 else 0
    
    stats = {
        'total_chunks': total_chunks,
        'total_output_frames': total_output_frames,
        'total_processing_frames': total_processing_frames,
        'chunks_with_context': chunks_with_context,
        'total_context_frames': total_context_frames,
        'processing_overhead': processing_overhead,
        'efficiency_ratio': efficiency_ratio,
        'average_chunk_size': total_output_frames / total_chunks if total_chunks > 0 else 0,
        'context_frame_percentage': (total_context_frames / total_processing_frames * 100) if total_processing_frames > 0 else 0
    }
    
    return stats


def optimize_context_parameters(total_frames: int, max_chunk_len: int, available_vram_gb: float = None) -> Tuple[int, int]:
    """
    Suggest optimized context parameters based on video characteristics.
    
    Args:
        total_frames: Total frames in video
        max_chunk_len: Desired maximum chunk length
        available_vram_gb: Available VRAM in GB (optional)
        
    Returns:
        Tuple of (optimized_max_chunk_len, recommended_context_overlap)
    """
    # Basic optimization rules
    optimized_chunk_len = max_chunk_len
    
    # Recommend context overlap as 25-50% of chunk length
    recommended_context = max_chunk_len // 4  # Start with 25%
    
    # Adjust based on video length
    if total_frames < 60:  # Short video
        recommended_context = min(recommended_context, 4)  # Smaller context for short videos
    elif total_frames > 300:  # Long video
        recommended_context = max(recommended_context, max_chunk_len // 3)  # Larger context for long videos
    
    # VRAM-based adjustments
    if available_vram_gb is not None:
        if available_vram_gb < 8:  # Low VRAM
            recommended_context = min(recommended_context, max_chunk_len // 6)
            optimized_chunk_len = min(optimized_chunk_len, 16)
        elif available_vram_gb > 16:  # High VRAM
            recommended_context = min(max_chunk_len // 2, recommended_context + 4)
    
    # Ensure context doesn't exceed chunk length
    recommended_context = min(recommended_context, optimized_chunk_len - 1)
    recommended_context = max(recommended_context, 0)
    
    return optimized_chunk_len, recommended_context


def process_context_based_chunks(
    chunks: List[Dict[str, int]],
    all_frames: List[Any],
    processing_function: callable,
    output_frames_dir: str,
    frame_files: List[str],
    logger,
    progress_callback: callable = None,
    **processing_kwargs
) -> Tuple[int, List[str]]:
    """
    Process all chunks using context-based sliding window approach.
    
    Args:
        chunks: List of chunk information dictionaries
        all_frames: List of all input frames
        processing_function: Function to process each chunk
        output_frames_dir: Directory to save output frames
        frame_files: List of frame filenames
        logger: Logger instance
        progress_callback: Optional progress callback function
        **processing_kwargs: Additional arguments for processing function
        
    Returns:
        Tuple of (total_processed_frames, list_of_saved_frame_paths)
    """
    total_processed_frames = 0
    saved_frame_paths = []
    total_chunks = len(chunks)
    
    logger.info(f"Starting context-based processing of {total_chunks} chunks")
    
    for i, chunk_info in enumerate(chunks):
        chunk_num = i + 1
        
        # Extract frames for processing (includes context if applicable)
        process_start = chunk_info['process_start_idx']
        process_end = chunk_info['process_end_idx']
        chunk_frames = all_frames[process_start:process_end]
        
        logger.info(f"Processing chunk {chunk_num}/{total_chunks}: "
                   f"frames {process_start+1}-{process_end} "
                   f"({len(chunk_frames)} frames, {chunk_info['context_frames']} context)")
        
        # Update progress
        if progress_callback:
            progress_callback(i / total_chunks, f"Processing chunk {chunk_num}/{total_chunks}")
        
        try:
            # Process the chunk
            processed_chunk = processing_function(chunk_frames, chunk_info, **processing_kwargs)
            
            # Extract only the output frames (excluding context)
            output_frames = extract_output_frames_from_context_chunk(processed_chunk, chunk_info)
            
            # Save the output frames
            output_start = chunk_info['output_start_idx']
            for j, frame in enumerate(output_frames):
                frame_index = output_start + j
                if frame_index < len(frame_files):
                    frame_filename = frame_files[frame_index]
                    frame_path = os.path.join(output_frames_dir, frame_filename)
                    
                    # Save frame (assuming frame is a numpy array or tensor)
                    save_frame(frame, frame_path)
                    saved_frame_paths.append(frame_path)
                    total_processed_frames += 1
            
            # Cleanup
            del processed_chunk, output_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {e}")
            raise
    
    logger.info(f"Context-based processing completed. Processed {total_processed_frames} frames.")
    return total_processed_frames, saved_frame_paths


def save_frame(frame, frame_path: str) -> None:
    """
    Save a frame to disk. Handles both numpy arrays and torch tensors.
    
    Args:
        frame: Frame data (numpy array or torch tensor)
        frame_path: Path to save the frame
    """
    import cv2
    
    # Convert torch tensor to numpy if needed
    if hasattr(frame, 'cpu'):  # torch tensor
        frame_np = frame.cpu().numpy()
    else:
        frame_np = frame
    
    # Ensure correct format for OpenCV (HWC, BGR)
    if len(frame_np.shape) == 3:
        # Convert RGB to BGR if needed
        if frame_np.shape[2] == 3:  # RGB image
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_np
    else:
        raise ValueError(f"Unexpected frame shape: {frame_np.shape}")
    
    # Save the frame
    cv2.imwrite(frame_path, frame_bgr)


def create_context_chunks_visualization(chunks: List[Dict[str, int]], total_frames: int) -> str:
    """
    Create a text visualization of how chunks overlap and process frames.
    
    Args:
        chunks: List of chunk information dictionaries
        total_frames: Total number of frames
        
    Returns:
        String visualization of the chunking strategy
    """
    if not chunks:
        return "No chunks to visualize"
    
    visualization = []
    visualization.append("Context-Based Sliding Window Visualization:")
    visualization.append("=" * 60)
    visualization.append("")
    
    # Create a timeline
    timeline_length = min(80, total_frames)
    scale_factor = total_frames / timeline_length
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        process_start = chunk['process_start_idx']
        process_end = chunk['process_end_idx']
        output_start = chunk['output_start_idx']
        output_end = chunk['output_end_idx']
        
        # Create visual representation
        line = ['.'] * timeline_length
        
        # Mark processing range
        start_pos = int(process_start / scale_factor)
        end_pos = int(process_end / scale_factor)
        for pos in range(max(0, start_pos), min(timeline_length, end_pos)):
            line[pos] = '-'
        
        # Mark output range
        start_pos = int(output_start / scale_factor)
        end_pos = int(output_end / scale_factor)
        for pos in range(max(0, start_pos), min(timeline_length, end_pos)):
            line[pos] = '#'
        
        # Mark context range
        if chunk['has_context']:
            context_end = output_start
            context_start = process_start
            start_pos = int(context_start / scale_factor)
            end_pos = int(context_end / scale_factor)
            for pos in range(max(0, start_pos), min(timeline_length, end_pos)):
                line[pos] = 'c'
        
        visualization.append(f"Chunk {chunk_num:2d}: {''.join(line)}")
        visualization.append(f"          Process: {process_start+1:3d}-{process_end:3d} | "
                           f"Output: {output_start+1:3d}-{output_end:3d} | "
                           f"Context: {chunk['context_frames']:2d}")
        visualization.append("")
    
    visualization.append("Legend:")
    visualization.append("  c = Context frames (input only)")
    visualization.append("  # = Output frames (processed and saved)")
    visualization.append("  - = Processing boundary")
    visualization.append("  . = Not processed in this chunk")
    
    return "\n".join(visualization)


# Example usage and testing functions
def test_context_chunks():
    """Test the context chunk calculation with various scenarios."""
    test_cases = [
        (73, 16, 8),   # User's example
        (100, 32, 8),  # Typical case
        (50, 16, 4),   # Shorter video
        (200, 24, 12), # Longer video
        (30, 16, 0),   # No context
    ]
    
    for total_frames, max_chunk, context in test_cases:
        print(f"\n=== Test Case: {total_frames} frames, {max_chunk} chunk size, {context} context ===")
        
        chunks = calculate_context_chunks(total_frames, max_chunk, context)
        stats = get_context_processing_stats(chunks)
        
        print(f"Chunks created: {len(chunks)}")
        print(f"Processing efficiency: {stats['efficiency_ratio']:.2%}")
        print(f"Context overhead: {stats['context_frame_percentage']:.1f}%")
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"  Chunk {i+1}: Process {chunk['process_start_idx']+1}-{chunk['process_end_idx']} "
                  f"→ Output {chunk['output_start_idx']+1}-{chunk['output_end_idx']}")
        
        if len(chunks) > 3:
            print(f"  ... and {len(chunks)-3} more chunks")


if __name__ == "__main__":
    # Run tests if module is executed directly
    test_context_chunks() 
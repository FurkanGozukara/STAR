"""
Chunk optimization module for better quality handling of last chunks.

Implements smart chunk boundaries to improve quality by ensuring the last chunk
has sufficient frames for optimal neural network processing.
"""

import math
import logging
from typing import Tuple, List, Dict


def calculate_optimized_chunk_boundaries(
    total_frames: int, 
    max_chunk_len: int,
    logger: logging.Logger = None
) -> List[Dict[str, int]]:
    """
    Calculate optimized chunk boundaries to ensure last chunk has optimal quality.
    
    Args:
        total_frames: Total number of frames in the video
        max_chunk_len: Maximum chunk length (user-specified chunk size)
        logger: Logger instance for debugging info
    
    Returns:
        List of dictionaries with chunk information:
        [
            {
                'chunk_idx': 0,
                'start_idx': 0,
                'end_idx': 32,
                'process_start_idx': 0,  # Same as start_idx for non-last chunks
                'process_end_idx': 32,   # Same as end_idx for non-last chunks
                'output_start_offset': 0,  # Offset in processed frames to extract
                'output_end_offset': 32,   # End offset in processed frames to extract
                'actual_output_frames': 32  # Number of frames to actually output
            },
            ...
        ]
    """
    if total_frames <= 0 or max_chunk_len <= 0:
        return []
    
    if total_frames <= max_chunk_len:
        # Single chunk, no optimization needed
        return [{
            'chunk_idx': 0,
            'start_idx': 0,
            'end_idx': total_frames,  # exclusive
            'process_start_idx': 0,
            'process_end_idx': total_frames,  # exclusive
            'output_start_offset': 0,
            'output_end_offset': total_frames,
            'actual_output_frames': total_frames
        }]
    
    # Calculate standard chunks
    standard_num_chunks = math.ceil(total_frames / max_chunk_len)
    last_chunk_start = (standard_num_chunks - 1) * max_chunk_len
    last_chunk_size = total_frames - last_chunk_start
    
    chunks = []
    
    # Check if last chunk needs optimization (trigger when last chunk is not equal to max_chunk_len)
    if last_chunk_size != max_chunk_len and standard_num_chunks > 1:
        if logger:
            logger.info(f"Optimizing chunks: Last chunk has {last_chunk_size} frames "
                       f"(not equal to {max_chunk_len} frames). Applying optimization.")
        
        # Create all chunks except the last one normally
        for chunk_idx in range(standard_num_chunks - 1):
            start_idx = chunk_idx * max_chunk_len
            end_idx = min((chunk_idx + 1) * max_chunk_len, total_frames)
            chunks.append({
                'chunk_idx': chunk_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,  # exclusive
                'process_start_idx': start_idx,
                'process_end_idx': end_idx,  # exclusive
                'output_start_offset': 0,
                'output_end_offset': end_idx - start_idx,
                'actual_output_frames': end_idx - start_idx
            })
        
        # Optimize the last chunk
        last_chunk_idx = standard_num_chunks - 1
        original_start = last_chunk_start
        original_end = total_frames  # exclusive
        
        # Calculate how many frames to extend backwards
        frames_needed = max_chunk_len
        extended_start = max(0, total_frames - frames_needed)
        extended_end = total_frames  # exclusive
        
        # Calculate which part of the processed result we need
        frames_to_skip = original_start - extended_start
        output_start_offset = frames_to_skip
        output_end_offset = frames_to_skip + last_chunk_size
        
        chunks.append({
            'chunk_idx': last_chunk_idx,
            'start_idx': original_start,  # for naming
            'end_idx': original_end,      # exclusive
            'process_start_idx': extended_start,  # inclusive
            'process_end_idx': extended_end,      # exclusive
            'output_start_offset': output_start_offset,
            'output_end_offset': output_end_offset,
            'actual_output_frames': last_chunk_size
        })
        
        if logger:
            logger.info(f"Last chunk optimization: Processing frames {extended_start}-{extended_end-1} "
                       f"but keeping only frames {output_start_offset}-{output_end_offset-1} "
                       f"(original {original_start}-{original_end-1})")
    
    else:
        # No optimization needed, create all chunks normally
        if logger and last_chunk_size != max_chunk_len:
            logger.info(f"Last chunk has {last_chunk_size} frames but only 1 chunk total, no optimization possible.")
        elif logger:
            logger.info(f"No chunk optimization needed: Last chunk has {last_chunk_size} frames "
                       f"(equal to {max_chunk_len} frames).")
        
        for chunk_idx in range(standard_num_chunks):
            start_idx = chunk_idx * max_chunk_len
            end_idx = min((chunk_idx + 1) * max_chunk_len, total_frames)
            # Ensure end_idx never exceeds the last valid frame index
            end_idx = min(end_idx, total_frames)
            chunks.append({
                'chunk_idx': chunk_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,  # exclusive
                'process_start_idx': start_idx,
                'process_end_idx': end_idx,  # exclusive
                'output_start_offset': 0,
                'output_end_offset': end_idx - start_idx,
                'actual_output_frames': end_idx - start_idx
            })
    
    return chunks


def get_chunk_frames_for_processing(
    all_frames_list: List, 
    chunk_info: Dict[str, int]
) -> List:
    """
    Extract the frames that need to be processed for a given chunk.
    
    Args:
        all_frames_list: List of all frame data (BGR frames, filenames, etc.)
        chunk_info: Chunk information dictionary from calculate_optimized_chunk_boundaries
        
    Returns:
        List of frames to process (may be more than output frames for optimized last chunk)
    """
    return all_frames_list[chunk_info['process_start_idx']:chunk_info['process_end_idx']]


def extract_output_frames_from_processed(
    processed_frames: List,
    chunk_info: Dict[str, int]
) -> List:
    """
    Extract the final output frames from processed chunk results.
    
    For normal chunks, this returns all processed frames.
    For optimized last chunks, this returns only the needed subset.
    
    Args:
        processed_frames: List of processed frame data
        chunk_info: Chunk information dictionary from calculate_optimized_chunk_boundaries
        
    Returns:
        List of frames to output (trimmed for optimized chunks)
    """
    start_offset = chunk_info['output_start_offset']
    end_offset = chunk_info['output_end_offset']
    return processed_frames[start_offset:end_offset]


def get_output_frame_names(
    all_frame_names: List[str],
    chunk_info: Dict[str, int]
) -> List[str]:
    """
    Get the frame names for the output frames (uses original indices).
    
    Args:
        all_frame_names: List of all frame filenames
        chunk_info: Chunk information dictionary
        
    Returns:
        List of frame names corresponding to output frames
    """
    start_idx = chunk_info['start_idx']
    end_idx = chunk_info['end_idx']
    
    # Ensure end_idx never exceeds the length of all_frame_names
    # This prevents IndexError when end_idx is set to total_frames
    max_valid_index = len(all_frame_names)
    end_idx = min(end_idx, max_valid_index)
    
    return all_frame_names[start_idx:end_idx]


def log_chunk_optimization_summary(
    chunks: List[Dict[str, int]], 
    total_frames: int,
    max_chunk_len: int,
    logger: logging.Logger = None
):
    """
    Log a summary of the chunk optimization results.
    
    Args:
        chunks: List of chunk information dictionaries
        total_frames: Total number of frames
        max_chunk_len: Maximum chunk length
        logger: Logger instance
    """
    if not logger:
        return
    
    optimized_chunks = [c for c in chunks if c['start_idx'] != c['process_start_idx']]
    
    logger.info(f"Chunk Optimization Summary:")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Max chunk length: {max_chunk_len}")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info(f"  Optimized chunks: {len(optimized_chunks)}")
    
    for chunk in chunks:
        if chunk['start_idx'] != chunk['process_start_idx']:
            logger.info(f"  Chunk {chunk['chunk_idx'] + 1}: "
                       f"Output frames {chunk['start_idx']}-{chunk['end_idx']-1} "
                       f"(processing {chunk['process_start_idx']}-{chunk['process_end_idx']-1}, "
                       f"keeping {chunk['output_start_offset']}-{chunk['output_end_offset']-1})")
        else:
            logger.info(f"  Chunk {chunk['chunk_idx'] + 1}: "
                       f"Frames {chunk['start_idx']}-{chunk['end_idx']-1} (normal)") 
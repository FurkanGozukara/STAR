"""
Context-based video processing module for STAR upscaling.

This module provides functionality to process video chunks with context overlap
for better temporal consistency, replacing the old sliding window approach.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import logging


def calculate_context_chunks(total_frames: int, max_chunk_len: int, context_overlap: int = 0) -> List[Dict[str, Any]]:
    """
    Calculate context-based chunk plan where each chunk (except first) includes
    previous frames as context but only outputs new frames.
    
    Args:
        total_frames: Total number of frames to process
        max_chunk_len: Maximum frames per chunk 
        context_overlap: Number of previous frames to include as context
        
    Returns:
        List of chunk dictionaries with keys:
        - 'process_start': First frame index to process (0-based, inclusive)
        - 'process_end': Last frame index to process (0-based, inclusive) 
        - 'output_start': First frame index to output (0-based, inclusive)
        - 'output_end': Last frame index to output (0-based, inclusive)
        - 'context_frames': Number of context frames included
        - 'new_frames': Number of new frames to output
        - 'total_frames': Total frames being processed in this chunk
    """
    if total_frames <= 0:
        return []
    
    if max_chunk_len <= 0:
        raise ValueError("max_chunk_len must be positive")
        
    if context_overlap < 0:
        raise ValueError("context_overlap cannot be negative")
        
    if context_overlap >= max_chunk_len:
        raise ValueError("context_overlap must be less than max_chunk_len")
    
    chunks = []
    current_output_start = 0
    
    while current_output_start < total_frames:
        # Calculate how many new frames we can output in this chunk
        remaining_frames = total_frames - current_output_start
        
        if len(chunks) == 0:
            # First chunk: no context, just process up to max_chunk_len frames
            frames_to_output = min(max_chunk_len, remaining_frames)
            process_start = 0
            process_end = frames_to_output - 1
            output_start = 0
            output_end = frames_to_output - 1
            context_frames = 0
            
        else:
            # Subsequent chunks: include context
            max_new_frames = max_chunk_len - context_overlap
            frames_to_output = min(max_new_frames, remaining_frames)
            
            # Calculate processing range
            process_start = max(0, current_output_start - context_overlap)
            process_end = current_output_start + frames_to_output - 1
            
            # Output range (only the new frames)
            output_start = current_output_start
            output_end = current_output_start + frames_to_output - 1
            
            context_frames = current_output_start - process_start
        
        # Create chunk info
        chunk_info = {
            'process_start': process_start,
            'process_end': process_end,
            'output_start': output_start, 
            'output_end': output_end,
            'context_frames': context_frames,
            'new_frames': frames_to_output,
            'total_frames': process_end - process_start + 1
        }
        
        chunks.append(chunk_info)
        current_output_start += frames_to_output
    
    return chunks


def get_chunk_frame_indices(chunk_info: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    Get the actual frame indices for processing and output from chunk info.
    
    Args:
        chunk_info: Chunk information dictionary from calculate_context_chunks
        
    Returns:
        Tuple of (process_indices, output_indices) where:
        - process_indices: List of 0-based frame indices to process
        - output_indices: List of 0-based frame indices to output (subset of process_indices)
    """
    process_indices = list(range(chunk_info['process_start'], chunk_info['process_end'] + 1))
    output_indices = list(range(chunk_info['output_start'], chunk_info['output_end'] + 1))
    
    return process_indices, output_indices


def validate_chunk_plan(chunks: List[Dict[str, Any]], total_frames: int) -> Tuple[bool, str]:
    """
    Validate that a chunk plan correctly covers all frames without gaps or overlaps in output.
    
    Args:
        chunks: List of chunk dictionaries
        total_frames: Expected total frames to cover
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not chunks:
        return False, "No chunks provided"
        
    if total_frames <= 0:
        return False, "Total frames must be positive"
    
    # Check that output frames cover all frames exactly once
    expected_outputs = set(range(total_frames))
    actual_outputs = set()
    
    for i, chunk in enumerate(chunks):
        # Validate chunk structure
        required_keys = ['process_start', 'process_end', 'output_start', 'output_end', 
                        'context_frames', 'new_frames', 'total_frames']
        for key in required_keys:
            if key not in chunk:
                return False, f"Chunk {i} missing key: {key}"
        
        # Validate chunk ranges
        if chunk['process_start'] > chunk['process_end']:
            return False, f"Chunk {i} has invalid process range: start > end"
            
        if chunk['output_start'] > chunk['output_end']:
            return False, f"Chunk {i} has invalid output range: start > end"
            
        # Check that output range is within process range
        if chunk['output_start'] < chunk['process_start'] or chunk['output_end'] > chunk['process_end']:
            return False, f"Chunk {i} output range not within process range"
        
        # Add output frames to our set
        chunk_outputs = set(range(chunk['output_start'], chunk['output_end'] + 1))
        
        # Check for overlaps
        overlap = actual_outputs.intersection(chunk_outputs) 
        if overlap:
            return False, f"Chunk {i} has overlapping output frames: {sorted(overlap)}"
            
        actual_outputs.update(chunk_outputs)
    
    # Check for gaps or missing frames
    missing_frames = expected_outputs - actual_outputs
    if missing_frames:
        return False, f"Missing output frames: {sorted(missing_frames)}"
        
    extra_frames = actual_outputs - expected_outputs
    if extra_frames:
        return False, f"Extra output frames: {sorted(extra_frames)}"
    
    return True, "Valid chunk plan"


def format_chunk_plan_summary(chunks: List[Dict[str, Any]], total_frames: int, 
                            max_chunk_len: int, context_overlap: int) -> str:
    """
    Format a human-readable summary of the chunk plan for logging.
    
    Args:
        chunks: List of chunk dictionaries
        total_frames: Total frames being processed
        max_chunk_len: Maximum chunk length used
        context_overlap: Context overlap used
        
    Returns:
        Formatted string summary
    """
    if not chunks:
        return "No chunks planned"
    
    lines = [
        f"Context Window Plan: {len(chunks)} chunks for {total_frames} frames",
        f"Settings: max_chunk_len={max_chunk_len}, context_overlap={context_overlap}"
    ]
    
    for i, chunk in enumerate(chunks):
        process_range = f"{chunk['process_start']}-{chunk['process_end']}"
        output_range = f"{chunk['output_start']}-{chunk['output_end']}"
        
        if chunk['context_frames'] > 0:
            context_info = f" (including {chunk['context_frames']} context frames)"
        else:
            context_info = ""
            
        lines.append(
            f"  Chunk {i+1}: Process frames {process_range}, "
            f"Output frames {output_range}{context_info}"
        )
    
    return "\n".join(lines)


def get_context_processing_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about the context processing plan.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with statistics:
        - total_chunks: Number of chunks
        - total_process_frames: Total frames that will be processed (including context)
        - total_output_frames: Total unique frames that will be output
        - context_overhead: Extra processing due to context (ratio)
        - avg_chunk_size: Average processing chunk size
        - avg_context_frames: Average context frames per chunk
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'total_process_frames': 0, 
            'total_output_frames': 0,
            'context_overhead': 0.0,
            'avg_chunk_size': 0.0,
            'avg_context_frames': 0.0
        }
    
    total_process_frames = sum(chunk['total_frames'] for chunk in chunks)
    total_output_frames = sum(chunk['new_frames'] for chunk in chunks)
    total_context_frames = sum(chunk['context_frames'] for chunk in chunks)
    
    context_overhead = (total_process_frames - total_output_frames) / total_output_frames if total_output_frames > 0 else 0.0
    avg_chunk_size = total_process_frames / len(chunks)
    avg_context_frames = total_context_frames / len(chunks)
    
    return {
        'total_chunks': len(chunks),
        'total_process_frames': total_process_frames,
        'total_output_frames': total_output_frames, 
        'context_overhead': context_overhead,
        'avg_chunk_size': avg_chunk_size,
        'avg_context_frames': avg_context_frames
    } 
"""
Corrected test for context sliding window logic - Updated to match user's exact specification
"""

def calculate_context_chunks_corrected(total_frames: int, max_chunk_len: int, context_overlap: int):
    """
    Calculate chunk boundaries matching the user's exact specification.
    
    Example with 16-frame chunks and 8-frame context for 126 total frames:
    - Chunk 1: Process frames 1-16, Output frames 1-16 (16 frames)
    - Chunk 2: Process frames 9-24, Output frames 17-24 (8 new frames, 8 context)
    - Chunk 3: Process frames 17-32, Output frames 25-32 (8 new frames, 8 context)
    - Chunk 4: Process frames 25-40, Output frames 33-40 (8 new frames, 8 context)
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


def test_user_specification():
    """Test the user's exact specification: 126 frames, 16 chunk size, 8 context."""
    chunks = calculate_context_chunks_corrected(126, 16, 8)
    
    print("=== User's Specification Test: 126 frames, 16 chunk size, 8 context ===")
    print(f"Total chunks: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        process_range = f"{chunk['process_start_idx']+1}-{chunk['process_end_idx']}"
        output_range = f"{chunk['output_start_idx']+1}-{chunk['output_end_idx']}"
        
        if chunk['has_context']:
            context_info = f" (with {chunk['context_frames']} context frames)"
        else:
            context_info = " (no context)"
        
        print(f"Chunk {chunk_num}: Process frames {process_range} ({chunk['process_length']} frames), "
              f"Output frames {output_range} ({chunk['output_length']} frames){context_info}")
    
    # Verify the expected pattern
    expected_outputs = [
        (1, 16, 16),   # Chunk 1: output 1-16 (16 frames)
        (17, 24, 8),   # Chunk 2: output 17-24 (8 frames)  
        (25, 32, 8),   # Chunk 3: output 25-32 (8 frames)
        (33, 40, 8),   # Chunk 4: output 33-40 (8 frames)
    ]
    
    print("\n=== Verification ===")
    for i, (expected_start, expected_end, expected_length) in enumerate(expected_outputs[:4]):
        if i < len(chunks):
            chunk = chunks[i]
            actual_start = chunk['output_start_idx'] + 1  # Convert to 1-based
            actual_end = chunk['output_end_idx']  # Already 1-based for end
            actual_length = chunk['output_length']
            
            match = (actual_start == expected_start and 
                    actual_end == expected_end and 
                    actual_length == expected_length)
            
            print(f"Chunk {i+1}: Expected {expected_start}-{expected_end} ({expected_length}), "
                  f"Got {actual_start}-{actual_end} ({actual_length}) - {'✓' if match else '✗'}")
        else:
            print(f"Chunk {i+1}: Expected {expected_start}-{expected_end} ({expected_length}), Got MISSING - ✗")


if __name__ == "__main__":
    test_user_specification() 
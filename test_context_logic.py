"""
Simple test for context sliding window logic (without torch dependencies)
"""

def calculate_context_chunks(total_frames: int, max_chunk_len: int, context_overlap: int):
    """Calculate chunk boundaries for context-based sliding window processing."""
    if total_frames <= 0 or max_chunk_len <= 0:
        return []
    
    # Ensure context_overlap doesn't exceed max_chunk_len
    effective_context_overlap = min(context_overlap, max_chunk_len)
    
    chunks = []
    current_output_start = 0
    chunk_index = 0
    
    while current_output_start < total_frames:
        # Calculate output range for this chunk
        output_start = current_output_start
        output_end = min(current_output_start + max_chunk_len, total_frames)
        
        # Calculate processing range (includes context)
        if chunk_index == 0:
            # First chunk: no context, process exactly what we output
            process_start = output_start
            process_end = output_end
        else:
            # Subsequent chunks: include context from previous frames
            # The processing should start from (output_start - context_overlap)
            # and end at output_end
            process_start = max(0, output_start - effective_context_overlap)
            process_end = output_end
        
        # Calculate offsets within the processed chunk
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
        current_output_start += max_chunk_len
        chunk_index += 1
    
    return chunks

def test_your_example():
    """Test your specific example: 73 frames, 16 chunk size, 8 context"""
    total_frames = 73
    max_chunk_len = 16
    context_overlap = 8
    
    print(f"=== YOUR EXAMPLE: {total_frames} frames, {max_chunk_len} chunk size, {context_overlap} context ===")
    
    chunks = calculate_context_chunks(total_frames, max_chunk_len, context_overlap)
    
    print(f"Total chunks created: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        
        print(f"Chunk {chunk_num}:")
        print(f"  Process frames: {chunk['process_start_idx']+1}-{chunk['process_end_idx']} ({chunk['process_length']} frames)")
        print(f"  Output frames:  {chunk['output_start_idx']+1}-{chunk['output_end_idx']} ({chunk['output_length']} frames)")
        
        if chunk['has_context']:
            print(f"  Context frames: {chunk['context_frames']} (frames {chunk['process_start_idx']+1}-{chunk['output_start_idx']})")
        else:
            print(f"  Context frames: 0 (first chunk)")
        print()
    
    # Verify the logic matches your description
    print("=== VERIFICATION ===")
    print("Expected behavior:")
    print("- First process: 1-16 (output 1-16)")
    print("- Second process: 8-24 (use only 17-24)")  
    print("- Third process: 16-32 (use only 25-32)")
    print()
    print("Actual behavior:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        chunk_num = i + 1
        if chunk_num == 1:
            print(f"- Chunk {chunk_num}: Process {chunk['process_start_idx']+1}-{chunk['process_end_idx']} (output {chunk['output_start_idx']+1}-{chunk['output_end_idx']})")
        else:
            print(f"- Chunk {chunk_num}: Process {chunk['process_start_idx']+1}-{chunk['process_end_idx']} (use only {chunk['output_start_idx']+1}-{chunk['output_end_idx']})")
    
    print()
    print("✅ Logic matches your specification!" if all([
        chunks[0]['process_start_idx'] == 0 and chunks[0]['process_end_idx'] == 16,  # 1-16
        chunks[1]['process_start_idx'] == 8 and chunks[1]['process_end_idx'] == 24,  # 8-24  
        chunks[1]['output_start_idx'] == 16 and chunks[1]['output_end_idx'] == 24,  # output 17-24
    ]) else "❌ Logic doesn't match")

if __name__ == "__main__":
    test_your_example() 
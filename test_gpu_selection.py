import torch
import os
import sys

# Add the parent directory to the path to import from secourses_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from secourses_app import get_available_gpus, set_gpu_device, get_gpu_device, SELECTED_GPU_ID

print("Testing GPU Selection...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # List available GPUs
    gpus = get_available_gpus()
    print(f"\nAvailable GPUs: {gpus}")
    
    # Test setting GPU 0
    print("\nTesting GPU 0 selection...")
    if len(gpus) > 0:
        result = set_gpu_device(gpus[0])
        print(f"Set result: {result}")
        print(f"Selected device: {get_gpu_device()}")
        print(f"SELECTED_GPU_ID: {SELECTED_GPU_ID}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Create a test tensor
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        test_tensor = test_tensor.to(get_gpu_device())
        print(f"Test tensor device: {test_tensor.device}")
        
        # Verify it's on the right GPU
        with torch.cuda.device(test_tensor.device):
            print(f"Tensor is on GPU: {torch.cuda.current_device()}")
    
    # Test setting GPU 1 if available
    if len(gpus) > 1:
        print("\nTesting GPU 1 selection...")
        result = set_gpu_device(gpus[1])
        print(f"Set result: {result}")
        print(f"Selected device: {get_gpu_device()}")
        print(f"SELECTED_GPU_ID: {SELECTED_GPU_ID}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Create another test tensor
        test_tensor2 = torch.tensor([4.0, 5.0, 6.0])
        test_tensor2 = test_tensor2.to(get_gpu_device())
        print(f"Test tensor device: {test_tensor2.device}")
        
        # Verify it's on the right GPU
        with torch.cuda.device(test_tensor2.device):
            print(f"Tensor is on GPU: {torch.cuda.current_device()}")
    
    # Test Auto mode
    print("\nTesting Auto mode...")
    result = set_gpu_device("Auto")
    print(f"Set result: {result}")
    print(f"Selected device: {get_gpu_device()}")
    print(f"SELECTED_GPU_ID: {SELECTED_GPU_ID}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Test invalid GPU ID
    print("\nTesting invalid GPU ID...")
    result = set_gpu_device(f"GPU {torch.cuda.device_count() + 1}: Fake GPU")
    print(f"Set result: {result}")
    print(f"Selected device still: {get_gpu_device()}")
    
else:
    print("No CUDA devices available for testing.")

print("\nTest complete.") 
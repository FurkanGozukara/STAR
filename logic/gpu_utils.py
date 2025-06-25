import torch
import logging

# Global variable to track selected GPU ID
SELECTED_GPU_ID = 0

def get_available_gpus():
    """Get list of available GPUs with their names and memory."""
    if not torch.cuda.is_available():
        return []

    gpu_count = torch.cuda.device_count()
    gpu_list = []

    for i in range(gpu_count):
        try:
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_name = gpu_props.name
            gpu_memory = gpu_props.total_memory / (1024 ** 3)
            gpu_list.append(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        except Exception as e:
            gpu_list.append(f"GPU {i}: Unknown")

    return gpu_list

def set_gpu_device(gpu_id, logger=None):
    """Set the GPU device to use for processing."""
    global SELECTED_GPU_ID
    try:
        if gpu_id is None:
            SELECTED_GPU_ID = 0
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.set_device(0)
            if logger:
                logger.info(f"GPU selection: Default mode - using GPU {SELECTED_GPU_ID}")
            return f"GPU selection: Default mode - using GPU {SELECTED_GPU_ID}"
        else:
            if isinstance(gpu_id, str) and gpu_id.startswith("GPU "):
                gpu_num = int(gpu_id.split(":")[0].replace("GPU ", "").strip())

                if torch.cuda.is_available() and 0 <= gpu_num < torch.cuda.device_count():
                    SELECTED_GPU_ID = gpu_num
                    torch.cuda.set_device(gpu_num)
                    if logger:
                        logger.info(f"GPU selection: Set to use GPU {gpu_num}")
                    return f"GPU selection: Set to use GPU {gpu_num}"
                else:
                    if logger:
                        logger.error(f"Invalid GPU ID {gpu_num}. Available GPUs: 0-{torch.cuda.device_count()-1}")
                    return f"Error: Invalid GPU ID {gpu_num}"
            else:
                gpu_num = int(gpu_id)
                if torch.cuda.is_available() and 0 <= gpu_num < torch.cuda.device_count():
                    SELECTED_GPU_ID = gpu_num
                    torch.cuda.set_device(gpu_num)
                    if logger:
                        logger.info(f"GPU selection: Set to use GPU {gpu_num}")
                    return f"GPU selection: Set to use GPU {gpu_num}"
                else:
                    if logger:
                        logger.error(f"Invalid GPU ID {gpu_num}")
                    return f"Error: Invalid GPU ID {gpu_num}"
    except Exception as e:
        error_msg = f"Error setting GPU device: {e}"
        if logger:
            logger.error(error_msg)
        return error_msg

def get_gpu_device(logger=None):
    """Get the current GPU device string."""
    global SELECTED_GPU_ID
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        if SELECTED_GPU_ID >= torch.cuda.device_count():
            if logger:
                logger.warning(f"Selected GPU {SELECTED_GPU_ID} is no longer available. Falling back to GPU 0.")
            SELECTED_GPU_ID = 0
        return f"cuda:{SELECTED_GPU_ID}"
    else:
        return "cpu"

def validate_gpu_availability():
    """Validate GPU availability and return status."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available. Running on CPU."

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return False, "No CUDA devices found. Running on CPU."

    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)

    return True, f"CUDA available with {device_count} GPU(s). Current: GPU {current_device} ({device_name})"

def get_selected_gpu_id():
    """Get the currently selected GPU ID."""
    global SELECTED_GPU_ID
    return SELECTED_GPU_ID

def set_selected_gpu_id(gpu_id):
    """Set the selected GPU ID directly."""
    global SELECTED_GPU_ID
    SELECTED_GPU_ID = gpu_id 
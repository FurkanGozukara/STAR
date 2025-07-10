#!/usr/bin/env python3
"""
Test script to validate CogVLM quantization fix.
This script tests the loading of CogVLM models with different quantization settings
to ensure the .to() error is resolved on Linux systems.
"""

import sys
import os
import torch
import logging

# Add the current directory to Python path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cogvlm_quantization():
    """Test CogVLM model loading with different quantization settings."""
    
    try:
        from logic.cogvlm_utils import COG_VLM_AVAILABLE, BITSANDBYTES_AVAILABLE, load_cogvlm_model, unload_cogvlm_model
        
        if not COG_VLM_AVAILABLE:
            logger.error("CogVLM components not available. Cannot run test.")
            return False
            
        logger.info(f"CogVLM Available: {COG_VLM_AVAILABLE}")
        logger.info(f"BitsAndBytes Available: {BITSANDBYTES_AVAILABLE}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA Device: {torch.cuda.current_device()}")
            logger.info(f"CUDA Device Name: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Capability: {torch.cuda.get_device_capability()}")
        
        # Test model path - adjust this to your actual model path
        model_paths = [
            "/workspace/STAR/models/cogvlm2-video-llama3-chat",  # Linux path
            "E:/Ultimate_Video_Processing_v1/STAR/models/cogvlm2-video-llama3-chat",  # Windows path
            "models/cogvlm2-video-llama3-chat"  # Relative path
        ]
        
        cogvlm_model_path = None
        for path in model_paths:
            if os.path.exists(path):
                cogvlm_model_path = path
                logger.info(f"Found CogVLM model at: {path}")
                break
        
        if not cogvlm_model_path:
            logger.error("CogVLM model not found at any expected paths:")
            for path in model_paths:
                logger.error(f"  - {path}")
            logger.error("Please update the model path in the test script.")
            return False
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test different quantization settings
        test_cases = []
        
        if device == 'cuda' and BITSANDBYTES_AVAILABLE:
            test_cases.extend([
                (4, "4-bit quantization"),
                (8, "8-bit quantization"),
            ])
        
        # Always test non-quantized
        test_cases.append((0, "No quantization (FP16/BF16)"))
        
        success_count = 0
        total_tests = len(test_cases)
        
        for quantization, description in test_cases:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {description}")
            logger.info(f"{'='*60}")
            
            try:
                # Test model loading
                logger.info(f"Loading model with quantization={quantization}, device={device}")
                model, tokenizer = load_cogvlm_model(quantization, device, cogvlm_model_path, logger)
                
                if model is not None and tokenizer is not None:
                    logger.info(f"✅ Successfully loaded {description}")
                    
                    # Test basic model properties
                    try:
                        first_param = next(model.parameters(), None)
                        if first_param:
                            logger.info(f"Model device: {first_param.device}")
                            logger.info(f"Model dtype: {first_param.dtype}")
                    except Exception as e:
                        logger.warning(f"Could not get model properties: {e}")
                    
                    # Unload the model
                    logger.info("Unloading model...")
                    unload_cogvlm_model('full', logger)
                    logger.info("Model unloaded successfully")
                    
                    success_count += 1
                else:
                    logger.error(f"❌ Failed to load {description}: model or tokenizer is None")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {description}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                
                # Try to clean up any partial state
                try:
                    unload_cogvlm_model('full', logger)
                except:
                    pass
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Successful tests: {success_count}/{total_tests}")
        
        if success_count == total_tests:
            logger.info("🎉 All tests passed! CogVLM quantization fix is working.")
            return True
        else:
            logger.warning(f"⚠️  {total_tests - success_count} test(s) failed.")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required dependencies are installed.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting CogVLM quantization fix test...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.warning("Transformers not available")
    
    try:
        import bitsandbytes
        logger.info(f"BitsAndBytes version: {bitsandbytes.__version__}")
    except ImportError:
        logger.warning("BitsAndBytes not available")
    
    success = test_cogvlm_quantization()
    
    if success:
        logger.info("\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
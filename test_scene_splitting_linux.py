#!/usr/bin/env python3
"""
Test script to verify scene splitting works on Linux with proper fallbacks.
This script tests the fixed scene splitting logic.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from logic.scene_utils import split_video_into_scenes
from logic.ffmpeg_utils import run_ffmpeg_command

def setup_logging():
    """Set up logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('test_scene_splitting.log')
        ]
    )
    return logging.getLogger(__name__)

def create_test_video(output_path, logger):
    """Create a simple test video for testing scene splitting."""
    logger.info(f"Creating test video: {output_path}")
    
    # Create a simple test video with color changes (should create multiple scenes)
    cmd = (
        f'ffmpeg -y -f lavfi -i "color=c=red:s=320x240:d=2" -f lavfi -i "color=c=blue:s=320x240:d=2" '
        f'-f lavfi -i "color=c=green:s=320x240:d=2" -filter_complex '
        f'"[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]" -map "[outv]" -r 25 -t 6 "{output_path}"'
    )
    
    try:
        result = run_ffmpeg_command(cmd, "Test Video Creation", logger, raise_on_error=False)
        if result and os.path.exists(output_path):
            logger.info(f"✅ Test video created successfully: {output_path}")
            return True
        else:
            logger.error(f"❌ Failed to create test video: {output_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Exception creating test video: {e}")
        return False

def test_scene_splitting(test_video_path, logger):
    """Test the scene splitting functionality."""
    logger.info("Starting scene splitting test...")
    
    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temp directory: {temp_dir}")
        
        # Test parameters - similar to what the app would use
        scene_split_params = {
            'split_mode': 'automatic',
            'min_scene_len': 0.5,  # Short for test
            'drop_short_scenes': False,
            'merge_last_scene': False,
            'frame_skip': 0,
            'threshold': 10.0,  # Lower threshold to detect color changes
            'min_content_val': 5.0,
            'frame_window': 2,
            'weights': [1.0, 1.0, 1.0, 0.0],
            'copy_streams': False,
            'use_mkvmerge': False,
            'rate_factor': 23,
            'preset': 'medium',
            'quiet_ffmpeg': False,
            'show_progress': False,
            'manual_split_type': 'duration',
            'manual_split_value': 2.0,
            'use_gpu': True  # Test GPU first, will fallback to CPU if needed
        }
        
        def progress_callback(progress_val, desc):
            logger.info(f"Progress: {progress_val:.1%} - {desc}")
        
        try:
            # Test the scene splitting
            scene_paths = split_video_into_scenes(
                test_video_path,
                temp_dir,
                scene_split_params,
                progress_callback,
                logger
            )
            
            logger.info(f"✅ Scene splitting completed successfully!")
            logger.info(f"✅ Number of scenes created: {len(scene_paths)}")
            
            # Verify that scene files were created
            valid_scenes = []
            for scene_path in scene_paths:
                if os.path.exists(scene_path) and os.path.getsize(scene_path) > 0:
                    valid_scenes.append(scene_path)
                    logger.info(f"✅ Valid scene: {os.path.basename(scene_path)} ({os.path.getsize(scene_path)} bytes)")
                else:
                    logger.warning(f"⚠️ Invalid scene: {scene_path}")
            
            if valid_scenes:
                logger.info(f"✅ Scene splitting test PASSED: {len(valid_scenes)} valid scenes created")
                return True
            else:
                logger.error(f"❌ Scene splitting test FAILED: No valid scenes created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Scene splitting test FAILED with exception: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            return False

def test_nvenc_availability(logger):
    """Test if NVENC is available on the system."""
    logger.info("Testing NVENC availability...")
    
    test_cmd = 'ffmpeg -loglevel error -f lavfi -i color=c=black:s=512x512:d=0.1:r=1 -c:v h264_nvenc -preset fast -f null -'
    
    try:
        result = run_ffmpeg_command(test_cmd, "NVENC Availability Test", logger, raise_on_error=False)
        if result:
            logger.info("✅ NVENC is available on this system")
            return True
        else:
            logger.info("ℹ️ NVENC is not available on this system (will use CPU encoding)")
            return False
    except Exception as e:
        logger.info(f"ℹ️ NVENC test failed: {e} (will use CPU encoding)")
        return False

def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("🧪 Starting Linux scene splitting compatibility test...")
    
    # Test NVENC availability first
    nvenc_available = test_nvenc_availability(logger)
    
    # Create test video
    test_video_path = "test_video_for_scene_splitting.mp4"
    
    try:
        if not create_test_video(test_video_path, logger):
            logger.error("❌ Failed to create test video. Cannot proceed with scene splitting test.")
            return False
        
        # Test scene splitting
        success = test_scene_splitting(test_video_path, logger)
        
        if success:
            logger.info("🎉 All tests PASSED! Scene splitting should work correctly on Linux.")
            return True
        else:
            logger.error("💥 Scene splitting test FAILED! Check the logs above for details.")
            return False
            
    finally:
        # Clean up test video
        if os.path.exists(test_video_path):
            try:
                os.remove(test_video_path)
                logger.info(f"🧹 Cleaned up test video: {test_video_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not clean up test video: {e}")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 
# logic/preset_handler.py
import os
import json
import logging
import dataclasses
from typing import Dict, Any, List, Optional

from .star_dataclasses import AppConfig # Import AppConfig for type hinting and default values

logger = logging.getLogger(__name__)

PRESETS_DIR = "presets"
LAST_PRESET_CONFIG_FILE = "last_preset.json"

def get_presets_dir() -> str:
    """Gets the absolute path to the presets directory, creating it if it doesn't exist."""
    base_dir = os.path.abspath(".")
    presets_path = os.path.join(base_dir, PRESETS_DIR)
    os.makedirs(presets_path, exist_ok=True)
    return presets_path

def get_preset_list() -> List[str]:
    """Returns a list of available preset filenames (without .json extension)."""
    presets_dir = get_presets_dir()
    try:
        files = [f for f in os.listdir(presets_dir) if f.endswith('.json')]
        preset_names = sorted([os.path.splitext(f)[0] for f in files])
        logger.debug(f"Available presets: {preset_names}")
        return preset_names
    except FileNotFoundError:
        logger.warning(f"Presets directory not found: {presets_dir}")
        return []

def save_last_used_preset_name(preset_name: str):
    """Saves the name of the last used preset to a config file."""
    presets_dir = get_presets_dir()
    config_path = os.path.join(presets_dir, LAST_PRESET_CONFIG_FILE)
    try:
        with open(config_path, 'w') as f:
            json.dump({"last_used_preset": preset_name}, f)
        logger.info(f"Set '{preset_name}' as the last used preset.")
    except Exception as e:
        logger.error(f"Could not save last used preset name: {e}")

def get_last_used_preset_name() -> Optional[str]:
    """Retrieves the name of the last used preset from the config file."""
    presets_dir = get_presets_dir()
    config_path = os.path.join(presets_dir, LAST_PRESET_CONFIG_FILE)
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
            return data.get("last_used_preset")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read last used preset file, it may be corrupted or missing: {e}")
        return None

def save_preset(app_config: AppConfig, preset_name: str) -> (bool, str):
    """Saves the current AppConfig to a JSON file."""
    if not preset_name or not preset_name.strip():
        return False, "Preset name cannot be empty."

    presets_dir = get_presets_dir()
    # Sanitize preset name to be a valid filename
    safe_preset_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_preset_name:
        return False, "Invalid preset name."
        
    filename = f"{safe_preset_name}.json"
    filepath = os.path.join(presets_dir, filename)

    try:
        # Convert AppConfig to a dictionary
        config_dict = dataclasses.asdict(app_config)
        
        # Ensure all sections are present in the saved preset
        # This is important for backward compatibility and complete preset saving
        required_sections = [
            'manual_comparison', 'standalone_face_restoration', 
            'preset_system', 'video_editing'
        ]
        
        for section in required_sections:
            if section not in config_dict:
                logger.warning(f"Section '{section}' missing from AppConfig, adding default values")
                if section == 'manual_comparison':
                    config_dict[section] = {
                        'video_count': 2,
                        'original_video': None,
                        'upscaled_video': None,
                        'third_video': None,
                        'fourth_video': None,
                        'layout': 'auto'
                    }
                elif section == 'standalone_face_restoration':
                    config_dict[section] = {
                        'fidelity_weight': 0.7,
                        'enable_colorization': False,
                        'batch_size': 1,
                        'save_frames': False,
                        'create_comparison': True,
                        'preserve_audio': True,
                        'input_video': None,
                        'mode': 'Single Video',
                        'batch_input_folder': '',
                        'batch_output_folder': '',
                        'codeformer_model': None
                    }
                elif section == 'preset_system':
                    config_dict[section] = {
                        'load_retries': 3,
                        'save_delay': 0.1,
                        'retry_delay': 0.2,
                        'load_delay': 0.1,
                        'conditional_updates_count': 20
                    }
                elif section == 'video_editing':
                    config_dict[section] = {
                        'precise_cutting_mode': 'precise',
                        'preview_first_segment': True
                    }
        
        # Remove fields that should not be in a preset
        # These are instance-specific and not part of a general configuration.
        config_dict.pop('paths', None)
        config_dict.pop('input_video_path', None)
        config_dict.pop('batch', None) # Batch folders are specific to a run
        if 'frame_folder' in config_dict:
             config_dict['frame_folder'].pop('input_path', None) # Don't save specific frame folder path

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        # Set this newly saved preset as the last used one
        save_last_used_preset_name(safe_preset_name)
        
        logger.info(f"Preset '{safe_preset_name}' saved to {filepath}")
        return True, f"Preset '{safe_preset_name}' saved successfully."
    except Exception as e:
        logger.error(f"Failed to save preset '{safe_preset_name}': {e}", exc_info=True)
        return False, f"Error saving preset: {e}"

def load_preset(preset_name: str) -> (Dict[str, Any], str):
    """Loads a preset from a JSON file and returns it as a dictionary."""
    if not preset_name:
        return None, "No preset selected."

    # Strip whitespace and sanitize the preset name
    preset_name = preset_name.strip()
    if not preset_name:
        return None, "Empty preset name."

    presets_dir = get_presets_dir()
    filepath = os.path.join(presets_dir, f"{preset_name}.json")

    # Debug logging to help identify the issue
    logger.debug(f"Attempting to load preset: '{preset_name}' from {filepath}")
    
    # Check if file exists and provide detailed error info
    if not os.path.exists(filepath):
        # List available files for debugging
        try:
            available_files = [f for f in os.listdir(presets_dir) if f.endswith('.json')]
            logger.debug(f"Available preset files in {presets_dir}: {available_files}")
        except Exception as e:
            logger.debug(f"Could not list preset directory: {e}")
        
        error_msg = f"Preset file not found: {preset_name}.json"
        # Use debug level instead of error for transient file not found issues
        # This reduces noise from race conditions during save/load operations
        logger.debug(error_msg + f" (Full path: {filepath})")
        return None, error_msg

    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Migration logic for backward compatibility
        config_dict = migrate_preset_data(config_dict)
        
        # Set this loaded preset as the last used one only if loading was successful
        save_last_used_preset_name(preset_name)
        
        logger.info(f"Preset '{preset_name}' loaded from {filepath}")
        return config_dict, f"Preset '{preset_name}' loaded successfully."
    except Exception as e:
        logger.error(f"Failed to load preset '{preset_name}': {e}", exc_info=True)
        return None, f"Error loading preset: {e}"

def migrate_preset_data(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old preset data to new format for backward compatibility."""
    # Handle choice name changes
    if 'resolution' in config_dict and 'target_res_mode' in config_dict['resolution']:
        old_mode = config_dict['resolution']['target_res_mode']
        # Migrate old choice name to new one
        if old_mode == 'Downscale then 4x':
            config_dict['resolution']['target_res_mode'] = 'Downscale then Upscale'
            logger.info(f"Migrated target_res_mode from '{old_mode}' to 'Downscale then Upscale'")
    
    # Add missing sections for new preset format
    if 'manual_comparison' not in config_dict:
        config_dict['manual_comparison'] = {
            'video_count': 2,
            'original_video': None,
            'upscaled_video': None,
            'third_video': None,
            'fourth_video': None,
            'layout': 'auto'
        }
        logger.info("Added missing manual_comparison section to preset")
    
    if 'standalone_face_restoration' not in config_dict:
        config_dict['standalone_face_restoration'] = {
            'fidelity_weight': 0.7,
            'enable_colorization': False,
            'batch_size': 1,
            'save_frames': False,
            'create_comparison': True,
            'preserve_audio': True,
            'input_video': None,
            'mode': 'Single Video',
            'batch_input_folder': '',
            'batch_output_folder': '',
            'codeformer_model': None
        }
        logger.info("Added missing standalone_face_restoration section to preset")
    else:
        # Ensure consistent values for string fields
        section = config_dict['standalone_face_restoration']
        if section.get('batch_input_folder') is None:
            section['batch_input_folder'] = ''
        if section.get('batch_output_folder') is None:
            section['batch_output_folder'] = ''
    
    if 'preset_system' not in config_dict:
        config_dict['preset_system'] = {
            'load_retries': 3,
            'save_delay': 0.1,
            'retry_delay': 0.2,
            'load_delay': 0.1,
            'conditional_updates_count': 20
        }
        logger.info("Added missing preset_system section to preset")
    
    if 'video_editing' not in config_dict:
        config_dict['video_editing'] = {
            'cutting_mode': 'time_ranges',
            'time_ranges_input': '',
            'frame_ranges_input': '',
            'precise_cutting_mode': 'precise',
            'preview_first_segment': True
        }
        logger.info("Added missing video_editing section to preset")
    else:
        # Add missing fields to existing video_editing section
        if 'cutting_mode' not in config_dict['video_editing']:
            config_dict['video_editing']['cutting_mode'] = 'time_ranges'
        if 'time_ranges_input' not in config_dict['video_editing']:
            config_dict['video_editing']['time_ranges_input'] = ''
        if 'frame_ranges_input' not in config_dict['video_editing']:
            config_dict['video_editing']['frame_ranges_input'] = ''
    
    # Add missing seedvr2 section if not present
    if 'seedvr2' not in config_dict:
        config_dict['seedvr2'] = {
            'enable': False,
            'model': None,
            'batch_size': 8,
            'temporal_overlap': 2,
            'quality_preset': 'quality',
            'use_gpu': True,
            'preserve_vram': False,
            'color_correction': True,
            'enable_frame_padding': True,
            'flash_attention': True,
            'scene_awareness': False,
            'consistency_validation': False,
            'chunk_optimization': False,
            'temporal_quality': 'balanced',
            'enable_multi_gpu': False,
            'gpu_devices': '0',
            'memory_optimization': 'auto',
            'enable_block_swap': False,
            'block_swap_counter': 0,
            'block_swap_offload_io': False,
            'block_swap_model_caching': True,
            'tiled_vae': False,
            'tile_size': (64, 64),
            'tile_stride': (32, 32),
            'cfg_scale': 1.0,
            'enable_chunk_preview': False,
            'chunk_preview_frames': 125,
            'keep_last_chunks': 5
        }
        logger.info("Added missing seedvr2 section to preset with all fields")
    else:
        # Ensure all fields exist in existing seedvr2 section
        seedvr2_defaults = {
            'enable': False,
            'model': None,
            'batch_size': 8,
            'temporal_overlap': 2,
            'quality_preset': 'quality',
            'use_gpu': True,
            'preserve_vram': False,
            'color_correction': True,
            'enable_frame_padding': True,
            'flash_attention': True,
            'scene_awareness': False,
            'consistency_validation': False,
            'chunk_optimization': False,
            'temporal_quality': 'balanced',
            'enable_multi_gpu': False,
            'gpu_devices': '0',
            'memory_optimization': 'auto',
            'enable_block_swap': False,
            'block_swap_counter': 0,
            'block_swap_offload_io': False,
            'block_swap_model_caching': True,
            'tiled_vae': False,
            'tile_size': (64, 64),
            'tile_stride': (32, 32),
            'cfg_scale': 1.0,
            'enable_chunk_preview': False,
            'chunk_preview_frames': 125,
            'keep_last_chunks': 5
        }
        
        # Add any missing fields
        for key, default_value in seedvr2_defaults.items():
            if key not in config_dict['seedvr2']:
                config_dict['seedvr2'][key] = default_value
                logger.info(f"Added missing seedvr2 field '{key}' with default value: {default_value}")
        
        # Handle model field migration
        if config_dict['seedvr2'].get('model') == "Model will be available soon":
            config_dict['seedvr2']['model'] = None
            logger.info("Migrated seedvr2 model from placeholder to null")
    
    # Add any other migrations here as needed in the future
    
    return config_dict
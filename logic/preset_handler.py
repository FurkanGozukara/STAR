# logic/preset_handler.py
import os
import json
import logging
import dataclasses
from typing import Dict, Any, List, Optional

from .dataclasses import AppConfig # Import AppConfig for type hinting and default values

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
    
    # Add any other migrations here as needed in the future
    
    return config_dict
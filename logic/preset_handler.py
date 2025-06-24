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
        return sorted([os.path.splitext(f)[0] for f in files])
    except FileNotFoundError:
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

    presets_dir = get_presets_dir()
    filepath = os.path.join(presets_dir, f"{preset_name}.json")

    if not os.path.exists(filepath):
        logger.error(f"Preset file not found: {filepath}")
        return None, f"Preset file not found: {preset_name}.json"

    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Set this loaded preset as the last used one
        save_last_used_preset_name(preset_name)
        
        logger.info(f"Preset '{preset_name}' loaded from {filepath}")
        return config_dict, f"Preset '{preset_name}' loaded successfully."
    except Exception as e:
        logger.error(f"Failed to load preset '{preset_name}': {e}", exc_info=True)
        return None, f"Error loading preset: {e}"
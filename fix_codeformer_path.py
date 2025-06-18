#!/usr/bin/env python3
"""
CodeFormer Model Path Fix Script

This script helps organize CodeFormer models in the correct directory structure
so that the STAR app uses them from pretrained_weight/ instead of downloading
to weights/ folder.

Usage:
    python fix_codeformer_path.py [--move] [--copy]
    
Options:
    --move: Move the model file instead of copying
    --copy: Copy the model file (default behavior)
    --scan: Only scan for existing models without moving
"""

import os
import sys
import shutil
import argparse
import glob
from pathlib import Path

def find_codeformer_models(search_dirs):
    """Find all CodeFormer model files in the specified directories."""
    found_models = []
    
    patterns = [
        "**/codeformer*.pth",
        "**/CodeFormer*.pth",
        "**/*codeformer*.pth"
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        print(f"Scanning: {search_dir}")
        
        for pattern in patterns:
            model_files = glob.glob(os.path.join(search_dir, pattern), recursive=True)
            for model_file in model_files:
                if os.path.isfile(model_file):
                    file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
                    found_models.append({
                        'path': model_file,
                        'name': os.path.basename(model_file),
                        'size_mb': file_size,
                        'directory': search_dir
                    })
                    print(f"  Found: {model_file} ({file_size:.1f} MB)")
    
    return found_models

def setup_pretrained_weight_structure(base_dir):
    """Create the expected directory structure in pretrained_weight."""
    pretrained_weight_dir = os.path.join(base_dir, 'pretrained_weight')
    codeformer_dir = os.path.join(pretrained_weight_dir, 'CodeFormer')
    
    os.makedirs(pretrained_weight_dir, exist_ok=True)
    os.makedirs(codeformer_dir, exist_ok=True)
    
    print(f"Created directory: {pretrained_weight_dir}")
    print(f"Created directory: {codeformer_dir}")
    
    return pretrained_weight_dir, codeformer_dir

def organize_models(found_models, target_dir, move_instead_of_copy=False):
    """Organize CodeFormer models in the target directory."""
    if not found_models:
        print("No CodeFormer models found to organize.")
        return
    
    print(f"\nOrganizing models in: {target_dir}")
    
    for model in found_models:
        source_path = model['path']
        target_path = os.path.join(target_dir, model['name'])
        
        # Check if target already exists
        if os.path.exists(target_path):
            print(f"  Target already exists: {target_path}")
            print(f"    Source: {source_path} ({model['size_mb']:.1f} MB)")
            print(f"    Target: {target_path} ({os.path.getsize(target_path) / (1024 * 1024):.1f} MB)")
            
            response = input("    Replace existing file? (y/N): ").strip().lower()
            if response != 'y':
                print("    Skipped.")
                continue
        
        try:
            if move_instead_of_copy:
                print(f"  Moving: {source_path} -> {target_path}")
                shutil.move(source_path, target_path)
            else:
                print(f"  Copying: {source_path} -> {target_path}")
                shutil.copy2(source_path, target_path)
            
            print(f"    Success! ({model['size_mb']:.1f} MB)")
            
        except Exception as e:
            print(f"    Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fix CodeFormer model paths for STAR app")
    parser.add_argument('--move', action='store_true', help='Move files instead of copying')
    parser.add_argument('--copy', action='store_true', help='Copy files (default)')
    parser.add_argument('--scan', action='store_true', help='Only scan for models without organizing')
    
    args = parser.parse_args()
    
    # Default to copy if neither move nor scan is specified
    if not args.move and not args.scan:
        args.copy = True
    
    # Get the base directory (STAR folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir  # This script is in STAR/ folder
    
    print("CodeFormer Model Path Fix Script")
    print("=" * 40)
    print(f"Base directory: {base_dir}")
    
    # Search locations for CodeFormer models
    search_dirs = [
        os.path.join(base_dir, 'weights'),  # Existing weights folder
        os.path.join(base_dir, '..', 'CodeFormer_STAR', 'weights'),  # CodeFormer_STAR weights
        os.path.join(base_dir, 'pretrained_weight'),  # Already in pretrained_weight
        os.path.join(base_dir, 'CodeFormer_STAR'),  # CodeFormer_STAR root
        os.path.join(base_dir, '..', 'CodeFormer_STAR'),  # Parent CodeFormer_STAR
    ]
    
    print("\nSearching for CodeFormer models...")
    found_models = find_codeformer_models(search_dirs)
    
    if not found_models:
        print("\nNo CodeFormer models found!")
        print("Make sure you have downloaded CodeFormer models first.")
        print("You can download them by:")
        print("1. Running the CodeFormer_STAR download script")
        print("2. Or manually downloading codeformer.pth from:")
        print("   https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth")
        return
    
    print(f"\nFound {len(found_models)} CodeFormer model(s):")
    for i, model in enumerate(found_models, 1):
        print(f"{i}. {model['name']} ({model['size_mb']:.1f} MB)")
        print(f"   Path: {model['path']}")
    
    if args.scan:
        print("\nScan complete. Use --copy or --move to organize models.")
        return
    
    # Setup target directory structure
    pretrained_weight_dir, codeformer_dir = setup_pretrained_weight_structure(base_dir)
    
    # Filter out models already in pretrained_weight to avoid duplicates
    models_to_organize = []
    for model in found_models:
        if not model['path'].startswith(pretrained_weight_dir):
            models_to_organize.append(model)
        else:
            print(f"Model already in pretrained_weight: {model['path']}")
    
    if not models_to_organize:
        print("\nAll models are already in the correct location!")
        return
    
    print(f"\nWill organize {len(models_to_organize)} model(s) to: {pretrained_weight_dir}")
    
    if args.move:
        print("Operation: MOVE (original files will be moved)")
    else:
        print("Operation: COPY (original files will be kept)")
    
    response = input("\nProceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Organize the models
    organize_models(models_to_organize, pretrained_weight_dir, args.move)
    
    print("\nDone! The STAR app should now use models from pretrained_weight/ directory.")
    print("You can verify this by running the app and checking the logs.")

if __name__ == "__main__":
    main() 
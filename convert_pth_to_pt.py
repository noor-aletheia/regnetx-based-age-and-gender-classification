#!/usr/bin/env python3
"""
Convert PyTorch .pth models to .pt format
Supports both individual files and batch conversion
"""

import torch
import os
import argparse
from pathlib import Path
import sys

def convert_pth_to_pt(pth_path: str, pt_path: str = None, keep_metadata: bool = True):
    """
    Convert a .pth file to .pt format
    
    Args:
        pth_path: Path to input .pth file
        pt_path: Path for output .pt file (optional, will auto-generate if None)
        keep_metadata: Whether to preserve training metadata (config, metrics, etc.)
    """
    pth_file = Path(pth_path)
    
    if not pth_file.exists():
        raise FileNotFoundError(f"Input file not found: {pth_path}")
    
    # Generate output path if not provided
    if pt_path is None:
        pt_path = pth_file.with_suffix('.pt')
    else:
        pt_path = Path(pt_path)
    
    print(f"Converting: {pth_file} -> {pt_path}")
    
    try:
        # Load the .pth file
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        if keep_metadata:
            # Keep everything as-is, just change extension
            torch.save(checkpoint, pt_path)
            print(f"✅ Converted with metadata preserved")
        else:
            # Extract only the model state dict for cleaner deployment
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                model_state = checkpoint['state_dict']
            else:
                # Assume the entire checkpoint is the state dict
                model_state = checkpoint
            
            torch.save(model_state, pt_path)
            print(f"✅ Converted (model weights only)")
        
        # Compare file sizes
        original_size = pth_file.stat().st_size / (1024 * 1024)  # MB
        new_size = pt_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"   Original: {original_size:.1f} MB")
        print(f"   Converted: {new_size:.1f} MB")
        
        return str(pt_path)
        
    except Exception as e:
        print(f"❌ Error converting {pth_file}: {e}")
        return None

def batch_convert_directory(input_dir: str, output_dir: str = None, keep_metadata: bool = True):
    """
    Convert all .pth files in a directory to .pt format
    
    Args:
        input_dir: Directory containing .pth files
        output_dir: Output directory (optional, will use input_dir if None)
        keep_metadata: Whether to preserve training metadata
    """
    input_path = Path(input_dir)
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .pth files
    pth_files = list(input_path.rglob("*.pth"))
    
    if not pth_files:
        print(f"No .pth files found in {input_dir}")
        return []
    
    print(f"Found {len(pth_files)} .pth files to convert...")
    
    converted_files = []
    for pth_file in pth_files:
        # Maintain directory structure
        relative_path = pth_file.relative_to(input_path)
        pt_file = output_path / relative_path.with_suffix('.pt')
        
        # Create output directory if needed
        pt_file.parent.mkdir(parents=True, exist_ok=True)
        
        result = convert_pth_to_pt(str(pth_file), str(pt_file), keep_metadata)
        if result:
            converted_files.append(result)
    
    return converted_files

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch .pth models to .pt format')
    parser.add_argument('input', nargs='?', help='Input .pth file or directory containing .pth files')
    parser.add_argument('--output', '-o', help='Output .pt file or directory')
    parser.add_argument('--batch', '-b', action='store_true', 
                       help='Batch convert all .pth files in input directory')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Save only model weights (no training metadata)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all .pth files in the workspace')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("=" * 60)
        print("AVAILABLE .pth MODELS IN WORKSPACE")
        print("=" * 60)
        
        pth_files = list(Path('/app').rglob("*.pth"))
        if pth_files:
            for pth_file in sorted(pth_files):
                size_mb = pth_file.stat().st_size / (1024 * 1024)
                print(f"{pth_file} ({size_mb:.1f} MB)")
        else:
            print("No .pth files found in /app")
        return
    
    if not args.input:
        parser.error("input is required unless using --list-models")
    
    keep_metadata = not args.no_metadata
    
    try:
        if args.batch or Path(args.input).is_dir():
            # Batch conversion
            converted = batch_convert_directory(args.input, args.output, keep_metadata)
            print(f"\n✅ Successfully converted {len(converted)} files")
            
            if converted:
                print("\nConverted files:")
                for file in converted:
                    print(f"  • {file}")
        else:
            # Single file conversion
            result = convert_pth_to_pt(args.input, args.output, keep_metadata)
            if result:
                print(f"\n✅ Successfully converted to: {result}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
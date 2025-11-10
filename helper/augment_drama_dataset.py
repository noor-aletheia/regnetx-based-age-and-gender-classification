#!/usr/bin/env python3
"""
Drama Dataset Augmentation Script
Applies strong augmentations to expand the drama dataset by 6x
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import random
from pathlib import Path
import argparse
from tqdm import tqdm

class DramaDatasetAugmenter:
    def __init__(self, input_dir, output_dir, csv_path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.csv_path = Path(csv_path)
        self.output_csv_path = self.output_dir / "groundtruth.csv"
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "female").mkdir(exist_ok=True)
        (self.output_dir / "male").mkdir(exist_ok=True)
        
        # Load original CSV
        self.df = pd.read_csv(self.csv_path)
        self.augmented_records = []
        
        # Augmentation parameters
        self.augmentations = {
            'horizontal_flip': {'prob': 0.5},
            'color_jitter': {
                'brightness': [-0.4, 0.4],
                'contrast': [-0.4, 0.4], 
                'saturation': [-0.4, 0.4]
            },
            'random_erasing': {'prob': 0.25, 'area_range': [0.02, 0.4]},
            'random_crop': {'scale_range': [0.8, 1.0]}
        }
    
    def horizontal_flip(self, image):
        """Apply horizontal flip augmentation"""
        return cv2.flip(image, 1)
    
    def color_jitter(self, image, brightness_factor, contrast_factor, saturation_factor):
        """Apply color jitter augmentation"""
        # Convert to PIL for easier color manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.0 + brightness_factor)
        
        # Apply contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.0 + contrast_factor)
        
        # Apply saturation
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.0 + saturation_factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def random_erasing(self, image, erase_prob=0.25):
        """Apply random erasing augmentation"""
        if random.random() > erase_prob:
            return image
        
        h, w, c = image.shape
        area = h * w
        
        # Random area between 2% and 40% of image
        target_area = random.uniform(0.02, 0.4) * area
        aspect_ratio = random.uniform(0.3, 3.3)
        
        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if h_erase < h and w_erase < w:
            x1 = random.randint(0, h - h_erase)
            y1 = random.randint(0, w - w_erase)
            
            # Fill with random values
            image_copy = image.copy()
            image_copy[x1:x1 + h_erase, y1:y1 + w_erase, :] = np.random.randint(0, 255, (h_erase, w_erase, c))
            return image_copy
        
        return image
    
    def random_crop_resize(self, image, scale_factor):
        """Apply random crop and resize back to original size"""
        h, w = image.shape[:2]
        
        # Calculate crop dimensions
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # Random crop position
        top = random.randint(0, h - new_h) if new_h < h else 0
        left = random.randint(0, w - new_w) if new_w < w else 0
        
        # Crop
        cropped = image[top:top + new_h, left:left + new_w]
        
        # Resize back to original dimensions
        return cv2.resize(cropped, (w, h))
    
    def generate_augmentation_combinations(self):
        """Generate all augmentation combinations for 6x expansion"""
        combinations = []
        
        # Original image
        combinations.append({
            'suffix': 'orig',
            'flip': False,
            'color_factors': (0.0, 0.0, 0.0),
            'erase': False,
            'crop_scale': 1.0
        })
        
        # Horizontal flip variant
        combinations.append({
            'suffix': 'flip',
            'flip': True,
            'color_factors': (0.0, 0.0, 0.0),
            'erase': False,
            'crop_scale': 1.0
        })
        
        # Color jitter variants (2 variations)
        combinations.append({
            'suffix': 'color1',
            'flip': False,
            'color_factors': (0.2, 0.2, 0.2),
            'erase': False,
            'crop_scale': 1.0
        })
        
        combinations.append({
            'suffix': 'color2',
            'flip': False,
            'color_factors': (-0.2, 0.3, -0.1),
            'erase': False,
            'crop_scale': 1.0
        })
        
        # Random erasing variant
        combinations.append({
            'suffix': 'erase',
            'flip': False,
            'color_factors': (0.0, 0.0, 0.0),
            'erase': True,
            'crop_scale': 1.0
        })
        
        # Random crop variant
        combinations.append({
            'suffix': 'crop',
            'flip': False,
            'color_factors': (0.0, 0.0, 0.0),
            'erase': False,
            'crop_scale': 0.85
        })
        
        return combinations
    
    def process_image(self, image_path, output_subdir, base_name, row):
        """Process a single image with all augmentations"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return
        
        combinations = self.generate_augmentation_combinations()
        
        for combo in combinations:
            # Start with original image
            augmented = image.copy()
            
            # Apply augmentations in sequence
            if combo['flip']:
                augmented = self.horizontal_flip(augmented)
            
            if combo['color_factors'] != (0.0, 0.0, 0.0):
                b, c, s = combo['color_factors']
                augmented = self.color_jitter(augmented, b, c, s)
            
            if combo['erase']:
                augmented = self.random_erasing(augmented)
            
            if combo['crop_scale'] != 1.0:
                augmented = self.random_crop_resize(augmented, combo['crop_scale'])
            
            # Generate output filename
            name_without_ext = Path(base_name).stem
            ext = Path(base_name).suffix
            output_name = f"{name_without_ext}_{combo['suffix']}{ext}"
            output_path = output_subdir / output_name
            
            # Save augmented image
            cv2.imwrite(str(output_path), augmented)
            
            # Create record for CSV
            record = {
                'name': output_name,
                'age': row['age'],
                'gender': row['gender'],
                'path': str(output_path)
            }
            self.augmented_records.append(record)
    
    def create_directory_structure(self):
        """Create output directory structure matching original"""
        for gender in ['female', 'male']:
            gender_dir = self.output_dir / gender
            gender_dir.mkdir(exist_ok=True)
            
            # Get unique age groups
            age_groups = self.df[self.df['gender'] == gender]['age'].unique()
            for age_group in age_groups:
                age_dir = gender_dir / f"age {age_group}"
                age_dir.mkdir(exist_ok=True)
    
    def augment_dataset(self):
        """Main function to augment entire dataset"""
        print("Creating directory structure...")
        self.create_directory_structure()
        
        print(f"Processing {len(self.df)} images with 6x augmentation...")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Augmenting images"):
            # Parse original path to get image location
            original_path = Path(row['path'])
            if not original_path.exists():
                # Try fixing the path by replacing /home/noor/Downloads/drama with input_dir
                fixed_path_str = str(row['path']).replace('/home/noor/Downloads/drama', str(self.input_dir))
                original_path = Path(fixed_path_str)
                if not original_path.exists():
                    # Try relative to input directory
                    original_path = self.input_dir / Path(row['path']).name
                    if not original_path.exists():
                        print(f"Warning: Image not found: {row['path']}")
                        continue
            
            # Determine output subdirectory
            gender = row['gender']
            age = row['age']
            output_subdir = self.output_dir / gender / f"age {age}"
            
            # Process image with all augmentations
            self.process_image(original_path, output_subdir, row['name'], row)
        
        # Save new CSV
        print("Saving augmented dataset CSV...")
        augmented_df = pd.DataFrame(self.augmented_records)
        augmented_df.to_csv(self.output_csv_path, index=False)
        
        print(f"Augmentation complete!")
        print(f"Original images: {len(self.df)}")
        print(f"Augmented images: {len(self.augmented_records)}")
        print(f"Expansion factor: {len(self.augmented_records) / len(self.df):.1f}x")
        print(f"Output directory: {self.output_dir}")
        print(f"New CSV: {self.output_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Augment drama dataset')
    parser.add_argument('--input_dir', type=str, default='/home/noor/Downloads/regnetx/drama',
                        help='Input drama dataset directory')
    parser.add_argument('--output_dir', type=str, default='/home/noor/Downloads/regnetx/drama_augmented',
                        help='Output directory for augmented dataset')
    parser.add_argument('--csv_path', type=str, default='/home/noor/Downloads/regnetx/drama/groundtruth.csv',
                        help='Path to original groundtruth CSV')
    
    args = parser.parse_args()
    
    # Verify input paths exist
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    if not Path(args.csv_path).exists():
        print(f"Error: CSV file {args.csv_path} does not exist")
        return
    
    # Create augmenter and run
    augmenter = DramaDatasetAugmenter(args.input_dir, args.output_dir, args.csv_path)
    augmenter.augment_dataset()

if __name__ == "__main__":
    main()
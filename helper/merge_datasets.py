#!/usr/bin/env python3
"""
Unified Dataset Merger
Intelligently merges /data and /drama_augmented into a unified dataset with proper train/test/val splits
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import random

class DatasetMerger:
    def __init__(self, data_dir, drama_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.drama_dir = Path(drama_dir)
        self.output_dir = Path(output_dir)
        
        # Split ratios for final dataset
        self.train_ratio = 0.8
        self.val_ratio = 0.1  
        self.test_ratio = 0.1
        
        # Create output structure
        self.create_output_structure()
        
        # Track statistics
        self.stats = {
            'original_data': {'train': 0, 'val': 0, 'test': 0},
            'drama_data': 0,
            'merged_data': {'train': 0, 'val': 0, 'test': 0},
            'total_images': 0
        }
        
    def create_output_structure(self):
        """Create unified output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        
    def normalize_age_format(self, age_str):
        """Normalize age format to consistent pattern"""
        age_str = str(age_str).strip()
        
        # Age mappings for consistency
        age_mappings = {
            '3-9': '3-9',
            '10-19': '10-19', 
            '20-29': '20-29',
            '30-39': '30-39',
            '40-49': '40-49',
            '50-59': '50-59',
            '60-69': '60-69',
            '70+': '70+'
        }
        
        return age_mappings.get(age_str, age_str)
    
    def normalize_gender_format(self, gender_str):
        """Normalize gender format to consistent pattern"""
        gender_str = str(gender_str).strip().lower()
        
        if gender_str in ['male', 'm']:
            return 'Male'
        elif gender_str in ['female', 'f']:
            return 'Female'
        else:
            return gender_str.title()
    
    def process_original_data(self):
        """Process original data directory and extract metadata"""
        print("Processing original data...")
        
        original_records = []
        
        # Process each split from original data
        for split in ['train', 'val', 'test']:
            csv_file = self.data_dir / split / f"{split}_groundtruth.csv"
            if not csv_file.exists():
                continue
                
            df = pd.read_csv(csv_file)
            self.stats['original_data'][split] = len(df)
            
            for _, row in df.iterrows():
                record = {
                    'filename': row['file'],
                    'age': self.normalize_age_format(row['age']),
                    'gender': self.normalize_gender_format(row['gender']),
                    'race': row.get('race', 'Unknown'),
                    'service_test': row.get('service_test', False),
                    'source_path': self.data_dir / split / row['file'],
                    'source_type': 'original',
                    'original_split': split
                }
                original_records.append(record)
        
        print(f"Original data: {len(original_records)} images")
        return original_records
    
    def process_drama_data(self):
        """Process drama augmented data and extract metadata"""
        print("Processing drama augmented data...")
        
        drama_records = []
        csv_file = self.drama_dir / "groundtruth.csv"
        
        if not csv_file.exists():
            print(f"Warning: Drama CSV not found at {csv_file}")
            return drama_records
            
        df = pd.read_csv(csv_file)
        self.stats['drama_data'] = len(df)
        
        for _, row in df.iterrows():
            # Extract actual file path from drama structure
            source_path = Path(str(row['path']).replace('/app/drama_augmented', str(self.drama_dir)))
            
            if source_path.exists():
                record = {
                    'filename': row['name'],
                    'age': self.normalize_age_format(row['age']),
                    'gender': self.normalize_gender_format(row['gender']),
                    'race': 'Unknown',  # Drama data doesn't have race info
                    'service_test': False,  # Drama data doesn't have service_test
                    'source_path': source_path,
                    'source_type': 'drama_augmented',
                    'original_split': None  # Will be assigned during unified split
                }
                drama_records.append(record)
        
        print(f"Drama augmented data: {len(drama_records)} images")
        return drama_records
    
    def create_stratified_split(self, all_records):
        """Create stratified train/val/test split maintaining age/gender balance"""
        print("Creating stratified split...")
        
        # Group by age and gender for stratification
        strata = defaultdict(list)
        for record in all_records:
            key = f"{record['age']}_{record['gender']}"
            strata[key].append(record)
        
        train_records = []
        val_records = []
        test_records = []
        
        # Split each stratum proportionally
        for key, records in strata.items():
            random.shuffle(records)  # Shuffle within stratum
            
            n_total = len(records)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            n_test = n_total - n_train - n_val  # Remainder goes to test
            
            train_records.extend(records[:n_train])
            val_records.extend(records[n_train:n_train + n_val])
            test_records.extend(records[n_train + n_val:])
            
        print(f"Split created - Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
        
        return train_records, val_records, test_records
    
    def copy_images_and_create_csv(self, records, split_name):
        """Copy images to output directory and create CSV"""
        print(f"Processing {split_name} split...")
        
        output_split_dir = self.output_dir / split_name
        csv_records = []
        
        for i, record in enumerate(tqdm(records, desc=f"Copying {split_name} images")):
            # Generate unique filename to avoid conflicts
            file_extension = record['source_path'].suffix
            unique_filename = f"{split_name}_{i+1:06d}{file_extension}"
            
            # Copy image to output directory
            dest_path = output_split_dir / unique_filename
            try:
                shutil.copy2(record['source_path'], dest_path)
                
                # Create CSV record
                csv_record = {
                    'file': unique_filename,
                    'age': record['age'],
                    'gender': record['gender'],
                    'race': record['race'],
                    'service_test': record['service_test'],
                    'source_type': record['source_type'],
                    'original_filename': record['filename']
                }
                csv_records.append(csv_record)
                
            except Exception as e:
                print(f"Error copying {record['source_path']}: {e}")
                continue
        
        # Create CSV file
        if csv_records:
            csv_df = pd.DataFrame(csv_records)
            csv_path = output_split_dir / f"{split_name}_groundtruth.csv"
            csv_df.to_csv(csv_path, index=False)
            print(f"Created {csv_path} with {len(csv_records)} records")
            
        self.stats['merged_data'][split_name] = len(csv_records)
        return len(csv_records)
    
    def generate_dataset_info(self):
        """Generate comprehensive dataset information"""
        info_content = f"""# Merged Dataset Information
Generated on: {pd.Timestamp.now()}

## Dataset Composition:
- **Original Data**: {sum(self.stats['original_data'].values())} images
  - Train: {self.stats['original_data']['train']}
  - Val: {self.stats['original_data']['val']} 
  - Test: {self.stats['original_data']['test']}
- **Drama Augmented**: {self.stats['drama_data']} images
- **Total Input**: {sum(self.stats['original_data'].values()) + self.stats['drama_data']} images

## Final Merged Dataset:
- **Train**: {self.stats['merged_data']['train']} images ({self.stats['merged_data']['train']/(sum(self.stats['merged_data'].values()))*100:.1f}%)
- **Val**: {self.stats['merged_data']['val']} images ({self.stats['merged_data']['val']/(sum(self.stats['merged_data'].values()))*100:.1f}%)
- **Test**: {self.stats['merged_data']['test']} images ({self.stats['merged_data']['test']/(sum(self.stats['merged_data'].values()))*100:.1f}%)
- **Total Output**: {sum(self.stats['merged_data'].values())} images

## CSV Format:
- `file`: Image filename in respective split directory
- `age`: Age group (3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+)
- `gender`: Gender (Male, Female)
- `race`: Race/ethnicity (from original data, 'Unknown' for drama data)
- `service_test`: Boolean service test flag (from original data, False for drama data)
- `source_type`: Data source ('original' or 'drama_augmented')
- `original_filename`: Original filename before renaming

## Directory Structure:
```
{self.output_dir.name}/
├── train/
│   ├── train_000001.jpg
│   ├── train_000002.jpg
│   ├── ...
│   └── train_groundtruth.csv
├── val/
│   ├── val_000001.jpg  
│   ├── val_000002.jpg
│   ├── ...
│   └── val_groundtruth.csv
├── test/
│   ├── test_000001.jpg
│   ├── test_000002.jpg
│   ├── ...
│   └── test_groundtruth.csv
└── dataset_info.md
```

## Merging Strategy:
1. **Unified Format**: Standardized age/gender labels across datasets
2. **Stratified Split**: Maintained age/gender distribution across splits  
3. **Conflict Resolution**: Unique filenames prevent overwrites
4. **Data Integration**: Combined original + augmented data with source tracking
5. **Quality Assurance**: Error handling and validation during copy process
"""
        
        info_path = self.output_dir / "dataset_info.md"
        with open(info_path, 'w') as f:
            f.write(info_content)
        print(f"Generated dataset info: {info_path}")
    
    def merge_datasets(self):
        """Main method to merge both datasets"""
        print("="*60)
        print("UNIFIED DATASET MERGER")
        print("="*60)
        
        # Set random seed for reproducible splits
        random.seed(42)
        np.random.seed(42)
        
        # Process both datasets
        original_records = self.process_original_data()
        drama_records = self.process_drama_data()
        
        # Combine all records
        all_records = original_records + drama_records
        self.stats['total_images'] = len(all_records)
        print(f"\nTotal combined images: {len(all_records)}")
        
        # Create stratified split
        train_records, val_records, test_records = self.create_stratified_split(all_records)
        
        # Copy images and create CSVs for each split
        print("\nCopying images and creating CSVs...")
        self.copy_images_and_create_csv(train_records, 'train')
        self.copy_images_and_create_csv(val_records, 'val') 
        self.copy_images_and_create_csv(test_records, 'test')
        
        # Generate dataset information
        self.generate_dataset_info()
        
        print("\n" + "="*60)
        print("MERGE COMPLETE!")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Train: {self.stats['merged_data']['train']} images")
        print(f"Val: {self.stats['merged_data']['val']} images") 
        print(f"Test: {self.stats['merged_data']['test']} images")
        print(f"Total: {sum(self.stats['merged_data'].values())} images")


def main():
    parser = argparse.ArgumentParser(description='Merge datasets intelligently')
    parser.add_argument('--data_dir', default='/home/noor/Downloads/regnetx/data',
                        help='Path to original data directory')
    parser.add_argument('--drama_dir', default='/home/noor/Downloads/regnetx/drama_augmented', 
                        help='Path to drama augmented directory')
    parser.add_argument('--output_dir', default='/home/noor/Downloads/regnetx/merged_dataset',
                        help='Path to output merged dataset directory')
    
    args = parser.parse_args()
    
    # Create merger and execute
    merger = DatasetMerger(args.data_dir, args.drama_dir, args.output_dir)
    merger.merge_datasets()


if __name__ == "__main__":
    main()
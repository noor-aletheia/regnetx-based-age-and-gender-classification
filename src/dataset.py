"""
Dataset processing and loading utilities with flexible CSV support
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union
import logging
from collections import Counter
import re
from config import Config

logger = logging.getLogger(__name__)


class ImageDimensionDetector:
    """Automatically detect image dimensions from dataset"""
    
    @staticmethod
    def detect_common_dimensions(image_paths: List[str], sample_size: int = 100) -> Tuple[int, int]:
        """
        Detect the most common image dimensions in the dataset
        
        Args:
            image_paths: List of image file paths
            sample_size: Number of images to sample for dimension detection
            
        Returns:
            Tuple of (width, height) representing most common dimensions
        """
        dimensions = []
        sample_paths = np.random.choice(image_paths, min(sample_size, len(image_paths)), replace=False)
        
        for img_path in sample_paths:
            try:
                with Image.open(img_path) as img:
                    dimensions.append(img.size)
            except Exception as e:
                logger.warning(f"Could not read image {img_path}: {e}")
                continue
        
        if not dimensions:
            logger.warning("Could not detect any image dimensions, using default 224x224")
            return (224, 224)
        
        # Find most common dimensions
        dimension_counts = Counter(dimensions)
        most_common_dim = dimension_counts.most_common(1)[0][0]
        
        logger.info(f"Detected most common image dimensions: {most_common_dim}")
        logger.info(f"Dimension distribution: {dict(dimension_counts.most_common(5))}")
        
        return most_common_dim


class CSVColumnDetector:
    """Automatically detect age and gender columns in CSV files"""
    
    @staticmethod
    def detect_age_column(df: pd.DataFrame) -> str:
        """Detect age column in DataFrame"""
        age_patterns = [
            r'^age$', r'^age_group$', r'^age_class$', r'^age_category$',
            r'^ages$', r'^age_label$', r'^age_range$', r'age', r'Age'
        ]
        
        for col in df.columns:
            for pattern in age_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    logger.info(f"Detected age column: {col}")
                    return col
        
        raise ValueError(f"Could not detect age column in CSV. Available columns: {list(df.columns)}")
    
    @staticmethod
    def detect_gender_column(df: pd.DataFrame) -> str:
        """Detect gender column in DataFrame"""
        gender_patterns = [
            r'^gender$', r'^sex$', r'^gender_class$', r'^gender_label$',
            r'^genders$', r'gender', r'Gender', r'sex', r'Sex'
        ]
        
        for col in df.columns:
            for pattern in gender_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    logger.info(f"Detected gender column: {col}")
                    return col
        
        raise ValueError(f"Could not detect gender column in CSV. Available columns: {list(df.columns)}")
    
    @staticmethod
    def detect_image_column(df: pd.DataFrame) -> str:
        """Detect image filename column in DataFrame"""
        image_patterns = [
            r'^image$', r'^filename$', r'^file$', r'^image_name$',
            r'^img$', r'^path$', r'^image_path$', r'image', r'filename'
        ]
        
        for col in df.columns:
            for pattern in image_patterns:
                if re.search(pattern, col, re.IGNORECASE):
                    logger.info(f"Detected image column: {col}")
                    return col
        
        # Fallback: use first column if no pattern matches
        logger.warning(f"Could not detect image column, using first column: {df.columns[0]}")
        return df.columns[0]


class LabelEncoder:
    """Encode age and gender labels to class indices"""
    
    def __init__(self):
        self.age_to_idx = {}
        self.gender_to_idx = {}
        self.idx_to_age = {}
        self.idx_to_gender = {}
    
    def fit_age_labels(self, age_labels: List[str]) -> Dict[str, int]:
        """Create age label to index mapping"""
        unique_ages = sorted(list(set(age_labels)))
        self.age_to_idx = {age: idx for idx, age in enumerate(unique_ages)}
        self.idx_to_age = {idx: age for age, idx in self.age_to_idx.items()}
        
        logger.info(f"Age classes ({len(unique_ages)}): {unique_ages}")
        return self.age_to_idx
    
    def fit_gender_labels(self, gender_labels: List[str]) -> Dict[str, int]:
        """Create gender label to index mapping"""
        unique_genders = sorted(list(set(gender_labels)))
        self.gender_to_idx = {gender: idx for idx, gender in enumerate(unique_genders)}
        self.idx_to_gender = {idx: gender for gender, idx in self.gender_to_idx.items()}
        
        logger.info(f"Gender classes ({len(unique_genders)}): {unique_genders}")
        return self.gender_to_idx
    
    def get_age_classes(self) -> int:
        """Get number of age classes"""
        return len(self.age_to_idx)
    
    def get_gender_classes(self) -> int:
        """Get number of gender classes"""
        return len(self.gender_to_idx)


class FaceDataset(Dataset):
    """Custom dataset for face age and gender classification"""
    
    def __init__(self, data: pd.DataFrame, root_dir: str, label_encoder: LabelEncoder, 
                 transform=None, target_transform=None):
        """
        Initialize the dataset
        
        Args:
            data: DataFrame containing image paths and labels
            root_dir: Root directory containing images
            label_encoder: LabelEncoder for converting labels to indices
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
        """
        self.data = data.reset_index(drop=True)
        self.root_dir = root_dir
        self.label_encoder = label_encoder
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path and load image
        row = self.data.iloc[idx]
        image_path = os.path.join(row['split_dir'], row['image'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get labels
        age_label = self.label_encoder.age_to_idx[row['age']]
        gender_label = self.label_encoder.gender_to_idx[row['gender']]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            age_label = self.target_transform(age_label)
            gender_label = self.target_transform(gender_label)
        
        return {
            'image': image,
            'age': torch.tensor(age_label, dtype=torch.long),
            'gender': torch.tensor(gender_label, dtype=torch.long),
            'image_path': image_path
        }



class DataProcessor:
    """Handle data loading, splitting, and preprocessing with flexible CSV support"""
    
    def __init__(self, config: Config):
        """
        Initialize data processor
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.root_dir = config.get('dataset.path')
        self.batch_size = config.get('dataset.batch_size', 64)
        self.num_workers = config.get('dataset.num_workers', 4)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Load data from train/val/test structure
        self.train_data, self.val_data, self.test_data = self._load_data_from_folders()
        
        # Detect image dimensions
        self.image_size = self._detect_image_dimensions()
        
        # Update config with detected image size
        self.config.set('dataset.detected_image_size', self.image_size)
        logger.info(f"Using image size: {self.image_size}x{self.image_size}")
        
        # Create transforms
        self.transforms = self._create_transforms()
        
    def _load_data_from_folders(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from train/val/test folder structure with CSV files"""
        
        def load_split_data(split_name: str) -> pd.DataFrame:
            split_dir = os.path.join(self.root_dir, split_name)
            # Try different CSV naming conventions
            csv_candidates = [
                'groundtruth.csv',
                f'{split_name}_groundtruth.csv',  # train_groundtruth.csv, val_groundtruth.csv
                f'{split_name}.csv'
            ]
            
            csv_path = None
            for csv_name in csv_candidates:
                potential_path = os.path.join(split_dir, csv_name)
                if os.path.exists(potential_path):
                    csv_path = potential_path
                    logger.info(f"Found CSV file: {csv_name} for {split_name} split")
                    break
            
            if csv_path is None:
                raise FileNotFoundError(f"No CSV file found in {split_dir}. Tried: {csv_candidates}")
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Groundtruth CSV not found: {csv_path}")
            
            logger.info(f"Loading {split_name} data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Detect column names
            image_col = CSVColumnDetector.detect_image_column(df)
            age_col = CSVColumnDetector.detect_age_column(df)
            gender_col = CSVColumnDetector.detect_gender_column(df)
            
            # Standardize column names
            df = df.rename(columns={
                image_col: 'image',
                age_col: 'age',
                gender_col: 'gender'
            })
            
            # Add split information
            df['split'] = split_name
            df['split_dir'] = split_dir
            
            # Optional fast validation (sample only)
            if len(df) > 1000:
                # For large datasets, only validate a sample
                sample_df = df.sample(min(100, len(df)), random_state=42)
                sample_valid = self._validate_image_files(sample_df, split_dir, sample_only=True)
                if len(sample_valid) < len(sample_df) * 0.95:  # Less than 95% valid
                    logger.warning(f"Sample validation failed for {split_name}, running full validation...")
                    df = self._validate_image_files(df, split_dir)
                else:
                    logger.info(f"Sample validation passed for {split_name} ({len(sample_valid)}/{len(sample_df)} valid)")
            else:
                # For small datasets, validate all
                df = self._validate_image_files(df, split_dir)
            
            logger.info(f"Loaded {len(df)} samples for {split_name}")
            self._print_split_statistics(df, split_name)
            
            return df
        
        # Load all splits
        train_data = load_split_data('train')
        val_data = load_split_data('val')
        
        # Test data is optional
        test_data = pd.DataFrame()
        test_dir = os.path.join(self.root_dir, 'test')
        if os.path.exists(test_dir):
            try:
                test_data = load_split_data('test')
            except FileNotFoundError:
                logger.warning("Test directory exists but no groundtruth.csv found")
        else:
            logger.warning("Test directory not found, will use validation data for testing")
            test_data = val_data.copy()
        
        # Fit label encoders on training data
        self.label_encoder.fit_age_labels(train_data['age'].tolist())
        self.label_encoder.fit_gender_labels(train_data['gender'].tolist())
        
        # Update config with detected classes
        self.config.set('dataset.age_classes', self.label_encoder.get_age_classes())
        self.config.set('dataset.gender_classes', self.label_encoder.get_gender_classes())
        
        return train_data, val_data, test_data
    
    def _validate_image_files(self, data: pd.DataFrame, split_dir: str, sample_only: bool = False) -> pd.DataFrame:
        """Validate that image files exist"""
        valid_indices = []
        
        for idx, row in data.iterrows():
            image_path = os.path.join(split_dir, row['image'])
            if os.path.exists(image_path):
                valid_indices.append(idx)
            else:
                logger.warning(f"Image file not found: {image_path}")
        
        valid_data = data.loc[valid_indices].reset_index(drop=True)
        
        if sample_only:
            logger.info(f"Sample validated {len(valid_data)}/{len(data)} image files")
        else:
            logger.info(f"Validated {len(valid_data)}/{len(data)} image files")
        
        return valid_data
    
    def _detect_image_dimensions(self) -> int:
        """Detect common image dimensions from dataset"""
        # Sample images from train set
        sample_images = []
        for _, row in self.train_data.head(50).iterrows():
            image_path = os.path.join(row['split_dir'], row['image'])
            sample_images.append(image_path)
        
        common_width, common_height = ImageDimensionDetector.detect_common_dimensions(sample_images)
        
        # Use the larger dimension as target size
        target_size = max(common_width, common_height)
        
        # Round to nearest common size
        if target_size <= 128:
            return 224  # Standard size for mobile networks
        elif target_size <= 256:
            return 256
        elif target_size <= 384:
            return 384
        else:
            return 512
    
    def _print_split_statistics(self, data: pd.DataFrame, split_name: str):
        """Print statistics for a data split"""
        logger.info(f"\n=== {split_name.upper()} Split Statistics ===")
        
        # Age distribution
        logger.info(f"\nAge Distribution:")
        age_counts = data['age'].value_counts().sort_index()
        for age, count in age_counts.items():
            percentage = (count / len(data)) * 100
            logger.info(f"  {age}: {count} ({percentage:.1f}%)")
        
        # Gender distribution
        logger.info(f"\nGender Distribution:")
        gender_counts = data['gender'].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(data)) * 100
            logger.info(f"  {gender}: {count} ({percentage:.1f}%)")
    
    def _create_transforms(self) -> Dict[str, transforms.Compose]:
        """Create image transforms for train, validation, and test sets"""
        # Get normalization parameters
        mean = self.config.get('augmentation.train.normalize.mean', [0.485, 0.456, 0.406])
        std = self.config.get('augmentation.train.normalize.std', [0.229, 0.224, 0.225])
        
        # Training transforms with augmentation
        train_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        # Add augmentations if specified
        if self.config.get('augmentation.train.horizontal_flip'):
            train_transforms.append(
                transforms.RandomHorizontalFlip(p=self.config.get('augmentation.train.horizontal_flip'))
            )
        
        if self.config.get('augmentation.train.rotation'):
            rotation_degrees = self.config.get('augmentation.train.rotation')
            train_transforms.append(
                transforms.RandomRotation(degrees=rotation_degrees)
            )
        
        if self.config.get('augmentation.train.color_jitter'):
            jitter_params = self.config.get('augmentation.train.color_jitter')
            train_transforms.append(
                transforms.ColorJitter(
                    brightness=jitter_params.get('brightness', 0),
                    contrast=jitter_params.get('contrast', 0),
                    saturation=jitter_params.get('saturation', 0),
                    hue=jitter_params.get('hue', 0)
                )
            )
        
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Validation/test transforms (no augmentation)
        val_test_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        
        return {
            'train': transforms.Compose(train_transforms),
            'val': transforms.Compose(val_test_transforms),
            'test': transforms.Compose(val_test_transforms)
        }
    
    def get_datasets(self) -> Tuple[FaceDataset, FaceDataset, FaceDataset]:
        """Get train, validation, and test datasets"""
        train_dataset = FaceDataset(
            self.train_data, 
            '', # Root dir handled in split_dir
            self.label_encoder,
            transform=self.transforms['train']
        )
        
        val_dataset = FaceDataset(
            self.val_data, 
            '',
            self.label_encoder,
            transform=self.transforms['val']
        )
        
        test_dataset = FaceDataset(
            self.test_data, 
            '',
            self.label_encoder,
            transform=self.transforms['test']
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test data loaders"""
        train_dataset, val_dataset, test_dataset = self.get_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate class weights for handling class imbalance, with optional power scaling"""
        # Configurable mode and alpha
        mode = self.config.get('class_weight_mode', 'default')
        alpha = float(self.config.get('class_weight_power_alpha', 1.0))

        # Age class weights
        age_counts = self.train_data['age'].value_counts()
        age_weights = torch.zeros(self.label_encoder.get_age_classes())
        total_age_samples = len(self.train_data)
        C_age = self.label_encoder.get_age_classes()
        for age, count in age_counts.items():
            age_idx = self.label_encoder.age_to_idx[age]
            if mode == 'power':
                weight = (total_age_samples / (C_age * count)) ** alpha
            else:
                weight = total_age_samples / (C_age * count)
            age_weights[age_idx] = weight

        # Gender class weights
        gender_counts = self.train_data['gender'].value_counts()
        gender_weights = torch.zeros(self.label_encoder.get_gender_classes())
        total_gender_samples = len(self.train_data)
        C_gender = self.label_encoder.get_gender_classes()
        for gender, count in gender_counts.items():
            gender_idx = self.label_encoder.gender_to_idx[gender]
            if mode == 'power':
                weight = (total_gender_samples / (C_gender * count)) ** alpha
            else:
                weight = total_gender_samples / (C_gender * count)
            gender_weights[gender_idx] = weight

        logger.info(f"Age class weights (mode={mode}, alpha={alpha}): {age_weights}")
        logger.info(f"Gender class weights (mode={mode}, alpha={alpha}): {gender_weights}")
        return age_weights, gender_weights
    
    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Get label to index mappings"""
        return self.label_encoder.age_to_idx, self.label_encoder.gender_to_idx
    
    def get_class_names(self) -> Tuple[List[str], List[str]]:
        """Get class names for age and gender"""
        age_names = [self.label_encoder.idx_to_age[i] for i in range(self.label_encoder.get_age_classes())]
        gender_names = [self.label_encoder.idx_to_gender[i] for i in range(self.label_encoder.get_gender_classes())]
        return age_names, gender_names
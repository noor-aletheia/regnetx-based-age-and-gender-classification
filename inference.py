import argparse
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

try:
    import sys
    sys.path.append('src')
    from config import Config
    from models import DualHeadRegNetX
except ImportError:
    print("Warning: Could not import from src/. Using standalone implementations.")
    
    class Config:
        def __init__(self, config_path: str = None):
            self.config = {
                'dataset': {'age_class_scheme': '4-class'},
                'augmentation': {
                    'val_test': {
                        'normalize': {
                            'mean': [0.498, 0.498, 0.498],
                            'std': [0.498, 0.498, 0.498]
                        }
                    }
                }
            }
        
        def get(self, key: str, default=None):
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from regnet import get_regnet, REGNET_MODEL_ZOO
    
    class DualHeadRegNetX(nn.Module):
        def __init__(self, model_name: str, age_classes: str = '4-class', pretrained: bool = True):
            super().__init__()
            self.model_name = model_name
            self.age_classes = age_classes
            if age_classes == '4-class':
                self.num_age_classes = 4
            elif age_classes == '8-class':
                self.num_age_classes = 8
            else:
                self.num_age_classes = 4
            self.num_gender_classes = 2
            model_mapping = {
                'regnet_1600m': 'regnet_1600m',
                'regnet_200m': 'regnet_200m',
                'regnet_400m': 'regnet_400m',
                'regnet_600m': 'regnet_600m',
                'regnet_800m': 'regnet_800m',
                'regnet_3200m': 'regnet_3200m',
                'regnet_6400m': 'regnet_6400m',
            }
            if model_name not in model_mapping:
                raise ValueError(f"Unsupported model: {model_name}")
            regnet_key = model_mapping[model_name]
            full_model = get_regnet(regnet_key, pretrained=pretrained)
            backbone_modules = []
            for name, module in full_model.named_children():
                if name != 'head':
                    backbone_modules.append(module)
            self.backbone = nn.Sequential(*backbone_modules)
            self.feature_dim = REGNET_MODEL_ZOO[regnet_key]['feature_dim'] if regnet_key in REGNET_MODEL_ZOO else 912
            self._init_heads()
        def _init_heads(self):
            self.age_head = nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, self.num_age_classes)
            )
            self.gender_head = nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, self.num_gender_classes)
            )
        def forward(self, x):
            features = self.backbone(x)
            if len(features.shape) == 2:
                age_logits = self.age_head[2:](features)
                gender_logits = self.gender_head[2:](features)
            else:
                age_logits = self.age_head(features)
                gender_logits = self.gender_head(features)
            return {
                'age': age_logits,
                'gender': gender_logits
            }


class SimpleDataset(Dataset):
    """Simple dataset for inference without complex augmentations"""
    
    def __init__(self, data_dir: str, age_classes: str = '4-class'):
        self.data_dir = Path(data_dir)
        self.age_classes = age_classes
        
        if age_classes == '4-class':
            self.age_class_names = ['0-9', '10-29', '30-49', '50-70+']
            self.age_ranges = [(0, 9), (10, 29), (30, 49), (50, 120)]
        elif age_classes == '8-class':
            self.age_class_names = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
            self.age_ranges = [(0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 120)]
        else:
            raise ValueError(f"Unsupported age classification scheme: {age_classes}")
        
        self.gender_class_names = ['Female', 'Male']
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.498, 0.498, 0.498], std=[0.498, 0.498, 0.498])
        ])
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load samples from dataset directory"""
        samples = []
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]  
            print(f"Loading dataset from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            
            required_cols = ['name', 'age', 'gender']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")
            
            for _, row in df.iterrows():
                image_path = self.data_dir / row['name']
                if not image_path.exists():
                    possible_paths = [
                        self.data_dir / 'test' / row['name'],
                        self.data_dir / 'val' / row['name'],
                        self.data_dir / 'train' / row['name'],
                        Path(row.get('path', '')) if 'path' in row else None
                    ]
                    
                    for path in possible_paths:
                        if path and path.exists():
                            image_path = path
                            break
                    else:
                        continue 
                
                age_class = self._age_range_to_class(row['age'])
                gender_class = 0 if row['gender'].lower() in ['female', 'f'] else 1
                
                samples.append({
                    'image_path': str(image_path),
                    'age': age_class,
                    'gender': gender_class,
                    'age_raw': row['age'],
                    'gender_raw': row['gender']
                })
        
        else:
            print("No CSV found. Attempting to infer from directory structure...")
            
            for split_dir in ['test', 'val', 'train', '.']:
                split_path = self.data_dir / split_dir if split_dir != '.' else self.data_dir
                if split_path.exists():
                    for img_path in split_path.glob("*.jpg"):
                        samples.append({
                            'image_path': str(img_path),
                            'age': -1,  # Unknown
                            'gender': -1,  # Unknown
                            'age_raw': 'Unknown',
                            'gender_raw': 'Unknown'
                        })
        
        if not samples:
            raise ValueError(f"No valid samples found in {self.data_dir}")
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def _age_to_class(self, age: int) -> int:
        """Convert numerical age to class index"""
        for i, (min_age, max_age) in enumerate(self.age_ranges):
            if min_age <= age <= max_age:
                return i
        return len(self.age_ranges) - 1 
    
    def _age_range_to_class(self, age_str) -> int:
        """Convert age range string (e.g., '10-29') or numerical age to class index"""
        if isinstance(age_str, str):
            age_str = str(age_str).strip()
            
            age_range_mapping = {
                '0-9': 0,
                '10-29': 1, 
                '30-49': 2,
                '50-70+': 3,
                '50+': 3,
                '70+': 3
            }
            
            if age_str in age_range_mapping:
                return age_range_mapping[age_str]
            
            if '-' in age_str:
                try:
                    parts = age_str.split('-')
                    min_age = int(parts[0])
                    return self._age_to_class(min_age)
                except:
                    pass
            
            try:
                age_num = int(age_str)
                return self._age_to_class(age_num)
            except:
                pass
        
        elif isinstance(age_str, (int, float)):
            return self._age_to_class(int(age_str))
        
        print(f"Warning: Could not parse age '{age_str}', defaulting to class 0")
        return 0
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        return {
            'image': image,
            'age': torch.tensor(sample['age'], dtype=torch.long),
            'gender': torch.tensor(sample['gender'], dtype=torch.long),
            'image_path': sample['image_path'],
            'age_raw': sample['age_raw'],
            'gender_raw': sample['gender_raw']
        }


class ModelInference:
    """Comprehensive model inference with detailed metrics"""
    
    def __init__(self, model_name: str, weights_path: str, age_classes: str = '4-class', device: str = "auto"):
        """
        Initialize inference engine
        
        Args:
            model_name: Name of the RegNetX model (e.g., 'regnetx_016')
            weights_path: Path to the trained weights (.pth file)
            age_classes: Age classification scheme ('4-class' or '8-class')
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.weights_path = Path(weights_path)
        self.age_classes = age_classes
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Weights: {weights_path}")
        print(f"Age classes: {age_classes}")
        
        self.model = self._load_model()
        self.model.eval()
        
        self.results = {}
        
    def _load_model(self) -> DualHeadRegNetX:
        """Load trained model from checkpoint"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        print(f"Loading weights from: {self.weights_path}")
        
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")
        
        if self.age_classes == '4-class':
            num_age_classes = 4
        elif self.age_classes == '8-class':
            num_age_classes = 8
        else:
            num_age_classes = 4
        
        model = DualHeadRegNetX(
            model_name=self.model_name,
            num_age_classes=num_age_classes,
            num_gender_classes=2,
            pretrained=False  
        )
        
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state dict: {e}")
        
        model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully. Parameters: {total_params:,}")
        
        return model
    
    def _create_dataloader(self, dataset_path: str, batch_size: int = 32, 
                          num_workers: int = 4) -> Tuple[DataLoader, List[str], List[str]]:
        """Create dataloader for inference dataset"""
        dataset = SimpleDataset(dataset_path, self.age_classes)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return dataloader, dataset.age_class_names, dataset.gender_class_names
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], task: str) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        valid_mask = (y_true >= 0) & (y_pred >= 0)
        if not np.any(valid_mask):
            print(f"Warning: No valid labels found for {task} classification")
            return {
                'accuracy': 0.0,
                'f1_weighted': 0.0,
                'confusion_matrix': np.zeros((len(class_names), len(class_names))),
                'classification_report': {},
                'class_names': class_names
            }
        
        y_true_filtered = y_true[valid_mask]
        y_pred_filtered = y_pred[valid_mask]
        
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=range(len(class_names)))
        
        report = classification_report(y_true_filtered, y_pred_filtered, 
                                     target_names=class_names, 
                                     output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names,
            'valid_samples': np.sum(valid_mask)
        }
    
    def _save_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                              task: str, output_dir: Path, accuracy: float):
        """Save confusion matrix as heatmap"""
        if cm.sum() == 0:
            print(f"Skipping confusion matrix for {task} (no valid data)")
            return
            
        plt.figure(figsize=(10, 8))
        
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title(f'{task.capitalize()} Classification Confusion Matrix\nAccuracy: {accuracy:.3f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        output_path = output_dir / f'confusion_matrix_{task}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix: {output_path}")
    
    def run_inference(self, dataset_path: str, output_dir: str = None, 
                     batch_size: int = 32, num_workers: int = 4, 
                     save_predictions: bool = False) -> Dict[str, Any]:
        """
        Run complete inference pipeline
        
        Args:
            dataset_path: Path to dataset directory
            output_dir: Directory to save results (optional)
            batch_size: Batch size for inference
            num_workers: Number of worker processes
            save_predictions: Whether to save individual predictions
            
        Returns:
            Dictionary containing all inference results
        """
        print(f"\n{'='*60}")
        print(f"STARTING INFERENCE")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Model: {self.model_name}")
        print(f"Weights: {self.weights_path}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        dataloader, age_classes, gender_classes = self._create_dataloader(
            dataset_path, batch_size, num_workers
        )
        
        print(f"Dataset loaded: {len(dataloader.dataset)} samples")
        print(f"Age classes: {age_classes}")
        print(f"Gender classes: {gender_classes}")
        
        all_age_preds = []
        all_gender_preds = []
        all_age_targets = []
        all_gender_targets = []
        all_image_paths = []
        all_predictions_data = []
        
        inference_times = []
        total_start_time = time.time()
        
        print("\nRunning inference...")
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                batch_start_time = time.time()
                
                images = batch['image'].to(self.device)
                age_labels = batch['age']
                gender_labels = batch['gender']
                
                outputs = self.model(images)
                
                age_pred = torch.argmax(outputs['age'], dim=1).cpu().numpy()
                gender_pred = torch.argmax(outputs['gender'], dim=1).cpu().numpy()
                
                age_probs = torch.softmax(outputs['age'], dim=1).cpu().numpy()
                gender_probs = torch.softmax(outputs['gender'], dim=1).cpu().numpy()

                def entropy(probs):
                    eps = 1e-8
                    return -np.sum(probs * np.log(probs + eps), axis=1)

                age_entropy = entropy(age_probs)
                gender_entropy = entropy(gender_probs)

                all_age_preds.extend(age_pred)
                all_gender_preds.extend(gender_pred)
                all_age_targets.extend(age_labels.numpy())
                all_gender_targets.extend(gender_labels.numpy())
                all_image_paths.extend(batch['image_path'])

                if save_predictions:
                    for i in range(len(age_pred)):
                        all_predictions_data.append({
                            'image_path': batch['image_path'][i],
                            'age_true': batch['age_raw'][i],
                            'gender_true': batch['gender_raw'][i],
                            'age_pred_class': age_classes[age_pred[i]],
                            'gender_pred_class': gender_classes[gender_pred[i]],
                            'age_confidence': float(age_probs[i][age_pred[i]]),
                            'gender_confidence': float(gender_probs[i][gender_pred[i]]),
                            'age_entropy': float(age_entropy[i]),
                            'gender_entropy': float(gender_entropy[i])
                        })

                batch_time = time.time() - batch_start_time
                inference_times.append(batch_time)
        
        total_time = time.time() - total_start_time
        
        all_age_preds = np.array(all_age_preds)
        all_gender_preds = np.array(all_gender_preds)
        all_age_targets = np.array(all_age_targets)
        all_gender_targets = np.array(all_gender_targets)
        
        print("\nCalculating metrics...")
        
        age_metrics = self._calculate_metrics(
            all_age_targets, all_age_preds, age_classes, 'age'
        )
        gender_metrics = self._calculate_metrics(
            all_gender_targets, all_gender_preds, gender_classes, 'gender'
        )
        
        total_samples = len(all_age_preds)
        avg_batch_time = np.mean(inference_times)
        fps = batch_size / avg_batch_time
        overall_fps = total_samples / total_time
        
        results = {
            'model_info': {
                'model_name': self.model_name,
                'weights_path': str(self.weights_path),
                'age_classes': age_classes,
                'gender_classes': gender_classes,
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'dataset_info': {
                'dataset_path': dataset_path,
                'total_samples': total_samples,
                'batch_size': batch_size,
                'valid_age_samples': age_metrics.get('valid_samples', 0),
                'valid_gender_samples': gender_metrics.get('valid_samples', 0)
            },
            'age_metrics': age_metrics,
            'gender_metrics': gender_metrics,
            'performance': {
                'total_inference_time': total_time,
                'avg_batch_time': avg_batch_time,
                'fps_per_batch': fps,
                'overall_fps': overall_fps,
                'samples_per_second': overall_fps
            }
        }
        
        if save_predictions:
            results['predictions'] = all_predictions_data
        
        self._print_results(results)
        
        if output_dir:
            self._save_results(results, output_dir, age_classes, gender_classes, save_predictions)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted results to console"""
        print(f"\n{'='*60}")
        print(f"INFERENCE RESULTS")
        print(f"{'='*60}")

        model_info = results['model_info']
        print(f"Model: {model_info['model_name']}")
        print(f"Parameters: {model_info['total_parameters']:,}")
        print(f"Age Classes: {len(model_info['age_classes'])}")
        print(f"Gender Classes: {len(model_info['gender_classes'])}")

        dataset_info = results['dataset_info']
        print(f"\nDataset: {dataset_info['total_samples']} samples")
        print(f"Valid age labels: {dataset_info['valid_age_samples']}")
        print(f"Valid gender labels: {dataset_info['valid_gender_samples']}")

        age_metrics = results['age_metrics']
        print(f"\nAge Classification:")
        print(f"  Accuracy: {age_metrics['accuracy']:.4f} ({age_metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Score: {age_metrics['f1_weighted']:.4f}")

        gender_metrics = results['gender_metrics']
        print(f"\nGender Classification:")
        print(f"  Accuracy: {gender_metrics['accuracy']:.4f} ({gender_metrics['accuracy']*100:.2f}%)")
        print(f"  F1 Score: {gender_metrics['f1_weighted']:.4f}")

        performance = results['performance']
        print(f"\nPerformance:")
        print(f"  Total Time: {performance['total_inference_time']:.2f}s")
        print(f"  FPS: {performance['overall_fps']:.2f} samples/second")
        print(f"  Avg Batch Time: {performance['avg_batch_time']:.4f}s")

        print(f"{'='*60}")

        if 'predictions' in results:
            try:
                import pandas as pd
                df = pd.DataFrame(results['predictions'])
                print("\n--- Gender-Age Intersectional Accuracy ---")
                gender_list = sorted(df['gender_true'].unique())
                age_list = sorted(df['age_true'].unique())
                for gender in gender_list:
                    for age in age_list:
                        group = df[(df['gender_true'] == gender) & (df['age_true'] == age)]
                        if len(group) > 0:
                            acc = (group['age_pred_class'] == group['age_true']).mean()
                            print(f"Gender: {gender:6} | Age: {age:8} | Accuracy: {acc:.3f} | N={len(group)}")
            except Exception as e:
                print(f"[Warning] Could not compute Gender-Age Intersectional Accuracy: {e}")
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path, 
                     age_classes: List[str], gender_classes: List[str], 
                     save_predictions: bool = False):
        """Save all results to files"""
        print(f"\nSaving results to: {output_dir}")
        
        self._save_confusion_matrix(
            results['age_metrics']['confusion_matrix'],
            age_classes,
            'age',
            output_dir,
            results['age_metrics']['accuracy']
        )
        
        self._save_confusion_matrix(
            results['gender_metrics']['confusion_matrix'],
            gender_classes,
            'gender',
            output_dir,
            results['gender_metrics']['accuracy']
        )
        
        results_path = output_dir / 'inference_results.json'
        json_results = self._prepare_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Saved detailed results: {results_path}")
        
        summary_path = output_dir / 'inference_summary.csv'
        summary_data = {
            'Model': [results['model_info']['model_name']],
            'Total_Samples': [results['dataset_info']['total_samples']],
            'Age_Accuracy': [results['age_metrics']['accuracy']],
            'Age_F1': [results['age_metrics']['f1_weighted']],
            'Gender_Accuracy': [results['gender_metrics']['accuracy']],
            'Gender_F1': [results['gender_metrics']['f1_weighted']],
            'FPS': [results['performance']['overall_fps']],
            'Inference_Time': [results['performance']['total_inference_time']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Saved summary: {summary_path}")
        
        if save_predictions and 'predictions' in results:
            predictions_path = output_dir / 'predictions.csv'
            predictions_df = pd.DataFrame(results['predictions'])
            predictions_df.to_csv(predictions_path, index=False)
            print(f"Saved predictions: {predictions_path}")
    
    def _prepare_for_json(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(
        description='RegNetX Model Inference - Single comprehensive script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python inference.py --model regnetx_016 --weights outputs/models/exp1/best.pth --dataset ./test_data
  
  # With custom output directory and batch size
  python inference.py --model regnetx_016 --weights model.pth --dataset ./data --output ./results --batch-size 64
  
  # Save individual predictions
  python inference.py --model regnetx_016 --weights model.pth --dataset ./data --save-predictions
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='RegNetX model name (e.g., regnetx_016, regnetx_008)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset directory')    
    parser.add_argument('--age-classes', type=str, choices=['4-class', '8-class'], 
                        default='4-class', help='Age classification scheme')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save results (default: ./inference_results)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes (default: 4)')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], 
                        default='auto', help='Device to use (default: auto)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions to CSV')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"❌ Error: Weights file not found: {args.weights}")
        return 1
    
    if not os.path.exists(args.dataset):
        print(f"❌ Error: Dataset directory not found: {args.dataset}")
        return 1
    
    if args.output is None:
        model_name = Path(args.weights).stem
        args.output = f"inference_results/{model_name}"
    
    try:
        inference = ModelInference(
            model_name=args.model,
            weights_path=args.weights,
            age_classes=args.age_classes,
            device=args.device
        )
        
        results = inference.run_inference(
            dataset_path=args.dataset,
            output_dir=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_predictions=args.save_predictions
        )
        
        print(f"\n✅ Inference completed successfully!")
        print(f"📊 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
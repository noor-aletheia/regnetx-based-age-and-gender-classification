import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
from config import Config
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class TrainingLogger:
    """Handle logging and metrics tracking"""
    
    def __init__(self, config: Config, model_name: str, experiment_name: str = None):
        """
        Initialize training logger
        
        Args:
            config: Configuration object
            model_name: Name of the model being trained
            experiment_name: Name of the experiment (for TensorBoard dir)
        """
        self.config = config
        self.model_name = model_name
        base_logs_dir = config.get('output.logs_dir')
        # Use experiment.model_name if set (for correct naming in multi-model runs)
        model_dir_name = config.get('experiment.model_name', model_name)
        # Only create per-model subdir for logs_dir, not for models_dir or tensorboard_dir
        models_list = config.get('models.variants', [])
        if isinstance(models_list, list) and len(models_list) > 1:
            self.logs_dir = os.path.join(base_logs_dir, model_dir_name)
        else:
            self.logs_dir = base_logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        self.history = []
        self.start_time = datetime.now()
        logger.info(f"Initialized logger for {model_name}")

        self.tb_writer = None
        if self.config.get('logging.enable_tensorboard', True):
            tb_base = self.config.get('output.tensorboard_dir', '/outputs/tensorboard')
            experiment = experiment_name or self.config.get('experiment_name', 'default_exp')
            # Do NOT add per-model subdir for tensorboard, keep as experiment/model_name
            tb_dir = os.path.join(tb_base, experiment, model_name)
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard SummaryWriter created at {tb_dir}")
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], phase: int):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            phase: Training phase (1 or 2)
        """
        log_entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.history.append(log_entry)

        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{k}/phase{phase}", v, epoch)
        
    
    def log_test_results(self, test_results: Dict[str, Any]):
        """Log test results"""
        age_metrics = test_results['age_metrics']
        gender_metrics = test_results['gender_metrics']
        
    
    
    def save_history(self):
        """Save training history to CSV"""
        if not self.history:
            return
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.logs_dir, 'training_history.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Training history saved to {csv_path}")
    
    def save_training_curves(self):
        """Save training curves as plots"""
        if not self.history:
            return
        df = pd.DataFrame(self.history)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation', marker='s')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(df['epoch'], df['train_age_accuracy'], label='Train', marker='o')
        axes[0, 1].plot(df['epoch'], df['val_age_accuracy'], label='Validation', marker='s')
        axes[0, 1].set_title('Age Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 2].plot(df['epoch'], df['train_gender_accuracy'], label='Train', marker='o')
        axes[0, 2].plot(df['epoch'], df['val_gender_accuracy'], label='Validation', marker='s')
        axes[0, 2].set_title('Gender Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        axes[1, 0].plot(df['epoch'], df['train_age_f1'], label='Train', marker='o')
        axes[1, 0].plot(df['epoch'], df['val_age_f1'], label='Validation', marker='s')
        axes[1, 0].set_title('Age F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 1].plot(df['epoch'], df['train_gender_f1'], label='Train', marker='o')
        axes[1, 1].plot(df['epoch'], df['val_gender_f1'], label='Validation', marker='s')
        axes[1, 1].set_title('Gender F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 2].plot(df['epoch'], df['learning_rate'], marker='o')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)
        for ax in axes.flat:
            phase_changes = df[df['phase'].diff() != 0]['epoch'].tolist()
            for epoch in phase_changes[1:]:
                ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.7, 
                          label='Phase Transition' if ax == axes[0, 0] else "")
        plt.tight_layout()
        plt.suptitle(f'Training Curves - {self.model_name}', y=1.02, fontsize=16)
        plot_path = os.path.join(self.logs_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {plot_path}")
    
    def close(self):
        """Close logger and save final results"""
        self.save_history()
        if self.config.get('logging.plot_training_curves', True):
            self.save_training_curves()
        
        
        end_time = datetime.now()
        training_time = (end_time - self.start_time).total_seconds()
        
        logger.info(f"Training completed for {self.model_name} in {training_time:.2f} seconds")

        if self.tb_writer:
            self.tb_writer.flush()
            self.tb_writer.close()
            logger.info("TensorBoard SummaryWriter closed.")


class ModelSaver:
    """Handle model saving and ONNX export"""
    
    def __init__(self, config: Config):
        """
        Initialize model saver
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models_dir = config.get('output.models_dir')
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int, metrics: Dict[str, float], model_name: str):
        """
        Save training checkpoint
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Current metrics
            model_name: Name of the model
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'model_name': model_name
        }
        checkpoint_path = os.path.join(self.models_dir, f'{model_name}_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    
    def save_best_model(self, model: torch.nn.Module, metrics: Dict[str, float], 
                       model_name: str, metric_name: str = 'val_loss'):
        """
        Save best model based on a specific metric
        Args:
            model: Model to save
            metrics: Model metrics
            model_name: Name of the model
            metric_name: Metric to use for determining best model
        """
        model_info = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'model_name': model_name,
            'model_architecture': str(model),
            'best_metric': metric_name,
            'best_metric_value': metrics.get(metric_name, 0)
        }
        model_path = os.path.join(self.models_dir, f'{model_name}_best.pth')
        torch.save(model_info, model_path)
        logger.info(f"Best model saved: {model_path}")
        return model_path


class ResultsAggregator:
    """Aggregate and compare results from multiple models"""
    
    def __init__(self, config: Config):
        """
        Initialize results aggregator
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logs_dir = config.get('output.logs_dir')
        self.results = []
    
    def add_model_results(self, model_name: str, test_results: Dict[str, Any],
                         training_time: float, model_info: Dict[str, Any], 
                         training_config: Dict[str, Any] = None, 
                         dataset_sizes: Dict[str, int] = None,
                         gflops: float = None, fps: float = None):
        """
        Add results from a trained model
        
        Args:
            model_name: Name of the model
            test_results: Test results dictionary
            training_time: Training time in seconds
            model_info: Model information
            training_config: Training configuration details
            dataset_sizes: Dataset split sizes
            gflops: Model computational complexity
            fps: Inference frames per second
        """
        age_metrics = test_results['age_metrics']
        gender_metrics = test_results['gender_metrics']
        
        if training_config is None:
            training_config = {}
            
        if dataset_sizes is None:
            dataset_sizes = {'train': 0, 'val': 0, 'test': 0}
        
        result = {
            'Model': model_name,
            'Phase 1 Epochs': training_config.get('phase1_epochs', 0),
            'Phase 2 Epochs': training_config.get('phase2_epochs', 0), 
            'Augmentation': training_config.get('augmentation_preset', 'unknown'),
            'LR (P1/P2)': f"{float(training_config.get('phase1_lr', 0)):.0e}/{float(training_config.get('phase2_lr', 0)):.0e}",
            'Weight Decay (P1/P2)': f"{float(training_config.get('phase1_wd', 0)):.0e}/{float(training_config.get('phase2_wd', 0)):.0e}",
            'scheduler': training_config.get('scheduler', 'unknown'),
            'Parameters': model_info.get('total_parameters', 0),
            'Training Time (s)': training_time,
            'Age Acc': age_metrics['accuracy'],
            'Age F1': age_metrics['f1'],
            'Gender Acc': gender_metrics['accuracy'],
            'Gender F1': gender_metrics['f1'],
            'Combined F1': age_metrics['f1'] + gender_metrics['f1'],
            'GFLOPs': gflops or 0.0,
            'class weight alpha': training_config.get('class_weight_alpha', 1.0),
            'cross entropy loss': training_config.get('loss_type', 'unknown'),
            'batch size': training_config.get('batch_size', 0),
            'fps': fps or 0.0,
            'train size': dataset_sizes.get('train', 0),
            'test size': dataset_sizes.get('test', 0),
            'val size': dataset_sizes.get('val', 0),
            'age class scheme': training_config.get('age_class_scheme', '8-class'),
        }
        
        self.results.append(result)
    
    def save_comparison_table(self):
        """Save comparison table of all models"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        df = df.sort_values('Combined F1', ascending=False)
        
        csv_path = os.path.join(self.logs_dir, 'comprehensive_training_results.csv')
        df.to_csv(csv_path, index=False)
        
        summary_path = os.path.join(self.logs_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== RegNetX Age & Gender Classification Results ===\n\n")
            f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: Mixed Drama + Internet (80/20 split)\n")
            f.write(f"Total training samples: {df.iloc[0]['train size'] if len(df) > 0 else 'N/A'}\n")
            f.write(f"Age class scheme: {df.iloc[0]['age class scheme'] if len(df) > 0 else 'N/A'}\n\n")
            
            f.write("Comprehensive Results (sorted by Combined F1 score):\n")
            f.write("="*120 + "\n")
            
            header = f"{'Model':<15} {'P1/P2 Epochs':<12} {'Augment':<20} {'LR(P1/P2)':<15} {'WD(P1/P2)':<15} {'Scheduler':<15} {'Params(M)':<10} {'Time(s)':<8} {'Age F1':<7} {'Gender F1':<9} {'Comb F1':<8} {'GFLOPs':<7}\n"
            f.write(header)
            f.write("-"*130 + "\n")
            
            for _, row in df.iterrows():
                params_m = row['Parameters'] / 1_000_000 if row['Parameters'] > 0 else 0
                line = f"{row['Model']:<15} {row['Phase 1 Epochs']}/{row['Phase 2 Epochs']:<8} {row['Augmentation']:<20} {row['LR (P1/P2)']:<15} {row['Weight Decay (P1/P2)']:<15} {row['scheduler']:<15} {params_m:<10.1f} {row['Training Time (s)']:<8.0f} {row['Age F1']:<7.3f} {row['Gender F1']:<9.3f} {row['Combined F1']:<8.3f} {row['GFLOPs']:<7.1f}\n"
                f.write(line)
            
            f.write("="*130 + "\n")
            
            if len(df) > 0:
                best_model = df.iloc[0]
                f.write(f"\n🏆 Best Model: {best_model['Model']}\n")
                f.write(f"   Combined F1: {best_model['Combined F1']:.4f}\n")
                f.write(f"   Age Accuracy: {best_model['Age Acc']:.4f} | Gender Accuracy: {best_model['Gender Acc']:.4f}\n")
                f.write(f"   Parameters: {best_model['Parameters']:,}\n")
                f.write(f"   Training Time: {best_model['Training Time (s)']:.1f}s\n")
                f.write(f"   Age Class Scheme: {best_model.get('age class scheme', 'N/A')}\n")
        
        logger.info(f"Model comparison saved to {csv_path}")
        logger.info(f"Training summary saved to {summary_path}")
        
        self._print_summary(df)
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary to console"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TRAINING SUMMARY")
        logger.info("="*80)
        
        for _, row in df.iterrows():
            logger.info(f"\n{row['Model']}:")
            logger.info(f"  Epochs: P1={row['Phase 1 Epochs']}, P2={row['Phase 2 Epochs']}")
            logger.info(f"  Age: Acc={row['Age Acc']:.4f}, F1={row['Age F1']:.4f}")
            logger.info(f"  Gender: Acc={row['Gender Acc']:.4f}, F1={row['Gender F1']:.4f}")
            logger.info(f"  Combined F1: {row['Combined F1']:.4f}")
            logger.info(f"  Parameters: {row['Parameters']:,}, GFLOPs: {row['GFLOPs']:.1f}")
            logger.info(f"  Training Time: {row['Training Time (s)']:.1f}s, FPS: {row['fps']:.1f}")
            logger.info(f"  Age Classes: {row.get('age class scheme', 'N/A')}")
        
        if len(df) > 0:
            best_model = df.iloc[0]
            logger.info(f"\n🏆 Best Model: {best_model['Model']}")
            logger.info(f"   Combined F1: {best_model['Combined F1']:.4f}")
            logger.info(f"   Dataset: Train={best_model['train size']}, Val={best_model['val size']}, Test={best_model['test size']}")
            logger.info(f"   Age Classes: {best_model.get('age class scheme', 'N/A')}")
        logger.info("="*80)
"""
Logging and metrics utilities for training
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import onnx
import onnxruntime as ort
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
from config import Config
# TensorBoard
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
        self.logs_dir = config.get('output.logs_dir')
        # Create directories
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Training history
        self.history = []
        
        # Start time
        self.start_time = datetime.now()
        
        logger.info(f"Initialized logger for {model_name}")

        # TensorBoard
        self.tb_writer = None
        if self.config.get('logging.enable_tensorboard', True):
            tb_base = self.config.get('output.tensorboard_dir', '/outputs/tensorboard')
            experiment = experiment_name or self.config.get('experiment_name', 'default_exp')
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
        # Add metadata
        log_entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Add to history
        self.history.append(log_entry)

        # Log to TensorBoard
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
        csv_path = os.path.join(self.logs_dir, f'{self.model_name}_training_history.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Training history saved to {csv_path}")
    
    def save_training_curves(self):
        """Save training curves as plots"""
        if not self.history:
            return
            
        df = pd.DataFrame(self.history)
        
        # Create training curves
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation', marker='s')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Age accuracy
        axes[0, 1].plot(df['epoch'], df['train_age_accuracy'], label='Train', marker='o')
        axes[0, 1].plot(df['epoch'], df['val_age_accuracy'], label='Validation', marker='s')
        axes[0, 1].set_title('Age Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gender accuracy
        axes[0, 2].plot(df['epoch'], df['train_gender_accuracy'], label='Train', marker='o')
        axes[0, 2].plot(df['epoch'], df['val_gender_accuracy'], label='Validation', marker='s')
        axes[0, 2].set_title('Gender Accuracy')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Age F1 score
        axes[1, 0].plot(df['epoch'], df['train_age_f1'], label='Train', marker='o')
        axes[1, 0].plot(df['epoch'], df['val_age_f1'], label='Validation', marker='s')
        axes[1, 0].set_title('Age F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Gender F1 score
        axes[1, 1].plot(df['epoch'], df['train_gender_f1'], label='Train', marker='o')
        axes[1, 1].plot(df['epoch'], df['val_gender_f1'], label='Validation', marker='s')
        axes[1, 1].set_title('Gender F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Learning rate
        axes[1, 2].plot(df['epoch'], df['learning_rate'], marker='o')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)
        
        # Add phase transitions
        for ax in axes.flat:
            phase_changes = df[df['phase'].diff() != 0]['epoch'].tolist()
            for epoch in phase_changes[1:]:  # Skip first epoch
                ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.7, 
                          label='Phase Transition' if ax == axes[0, 0] else "")
        
        plt.tight_layout()
        plt.suptitle(f'Training Curves - {self.model_name}', y=1.02, fontsize=16)
        
        # Save plot
        plot_path = os.path.join(self.logs_dir, f'{self.model_name}_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_path}")
    
    def close(self):
        """Close logger and save final results"""
        self.save_history()
        if self.config.get('logging.plot_training_curves', True):
            self.save_training_curves()
        
        
        # Calculate total training time
        end_time = datetime.now()
        training_time = (end_time - self.start_time).total_seconds()
        
        logger.info(f"Training completed for {self.model_name} in {training_time:.2f} seconds")

        # Close TensorBoard writer
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
    
    def export_to_onnx(self, model: torch.nn.Module, model_name: str, 
                      input_shape: tuple = (1, 3, 224, 224)):
        """
        Export model to ONNX format (FP32 and FP16 versions)
        
        Args:
            model: Model to export
            model_name: Name of the model
            input_shape: Input tensor shape
        """
        if not self.config.get('logging.export_onnx', True):
            return
            
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Export to ONNX (FP32)
        onnx_path = os.path.join(self.models_dir, f'{model_name}.onnx')
        onnx_fp16_path = os.path.join(self.models_dir, f'{model_name}_fp16.onnx')
        
        try:
            # Export FP32 ONNX model
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['age_output', 'gender_output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'age_output': {0: 'batch_size'},
                    'gender_output': {0: 'batch_size'}
                }
            )
            
            # Verify FP32 ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test FP32 with ONNX Runtime
            ort_session = ort.InferenceSession(onnx_path)
            test_input = dummy_input.cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            logger.info(f"✅ FP32 ONNX model exported and verified: {onnx_path}")
            
            # Convert to FP16 (if enabled)
            if self.config.get('logging.export_onnx_fp16', True):
                try:
                    from onnxconverter_common import float16
                    
                    # Load FP32 model and convert to FP16
                    onnx_model_fp32 = onnx.load(onnx_path)
                    onnx_model_fp16 = float16.convert_float_to_float16(onnx_model_fp32)
                    
                    # Save FP16 model
                    onnx.save(onnx_model_fp16, onnx_fp16_path)
                    
                    # Verify FP16 model
                    onnx.checker.check_model(onnx_model_fp16)
                    
                    # Test FP16 with ONNX Runtime (if available)
                    try:
                        ort_session_fp16 = ort.InferenceSession(onnx_fp16_path)
                        test_input_fp16 = test_input.astype(np.float16) if test_input.dtype == np.float32 else test_input
                        ort_inputs_fp16 = {ort_session_fp16.get_inputs()[0].name: test_input_fp16}
                        ort_outputs_fp16 = ort_session_fp16.run(None, ort_inputs_fp16)
                        
                        logger.info(f"✅ FP16 ONNX model exported and verified: {onnx_fp16_path}")
                        
                        # Compare model sizes (including external data files)
                        def get_total_onnx_size(onnx_file_path):
                            total_size = os.path.getsize(onnx_file_path)
                            # Check for external data file
                            data_file = onnx_file_path + '.data'
                            if os.path.exists(data_file):
                                total_size += os.path.getsize(data_file)
                            return total_size / (1024 * 1024)  # MB
                        
                        fp32_size = get_total_onnx_size(onnx_path)
                        fp16_size = get_total_onnx_size(onnx_fp16_path)
                        compression_ratio = fp32_size / fp16_size if fp16_size > 0 else 0
                        
                        logger.info(f"📊 Model sizes - FP32: {fp32_size:.2f}MB, FP16: {fp16_size:.2f}MB (Compression: {compression_ratio:.2f}x)")
                        
                    except Exception as fp16_test_error:
                        logger.warning(f"FP16 ONNX model created but testing failed: {fp16_test_error}")
                        logger.info(f"✅ FP16 ONNX model exported: {onnx_fp16_path}")
                        
                except ImportError:
                    logger.warning("onnxconverter-common not available. Install with: pip install onnxconverter-common")
                    logger.info("Only FP32 ONNX model was exported.")
                except Exception as fp16_error:
                    logger.error(f"Failed to create FP16 ONNX model: {fp16_error}")
            
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")


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
                         training_time: float, model_info: Dict[str, Any]):
        """
        Add results from a trained model
        
        Args:
            model_name: Name of the model
            test_results: Test results dictionary
            training_time: Training time in seconds
            model_info: Model information
        """
        age_metrics = test_results['age_metrics']
        gender_metrics = test_results['gender_metrics']
        
        result = {
            'model_name': model_name,
            'total_parameters': model_info.get('total_parameters', 0),
            'trainable_parameters': model_info.get('trainable_parameters', 0),
            'training_time_seconds': training_time,
            'age_accuracy': age_metrics['accuracy'],
            'age_precision': age_metrics['precision'],
            'age_recall': age_metrics['recall'],
            'age_f1': age_metrics['f1'],
            'gender_accuracy': gender_metrics['accuracy'],
            'gender_precision': gender_metrics['precision'],
            'gender_recall': gender_metrics['recall'],
            'gender_f1': gender_metrics['f1']
        }
        
        self.results.append(result)
    
    def save_comparison_table(self):
        """Save comparison table of all models"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Sort by combined performance (age F1 + gender F1)
        df['combined_f1'] = df['age_f1'] + df['gender_f1']
        df = df.sort_values('combined_f1', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(self.logs_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Create a formatted summary
        summary_path = os.path.join(self.logs_dir, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== MobileNet Age & Gender Classification Results ===\n\n")
            f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Model Comparison (sorted by combined F1 score):\n")
            f.write("-" * 80 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"Model: {row['model_name']}\n")
                f.write(f"  Parameters: {row['total_parameters']:,} (trainable: {row['trainable_parameters']:,})\n")
                f.write(f"  Training Time: {row['training_time_seconds']:.1f}s\n")
                f.write(f"  Age - Acc: {row['age_accuracy']:.4f}, F1: {row['age_f1']:.4f}\n")
                f.write(f"  Gender - Acc: {row['gender_accuracy']:.4f}, F1: {row['gender_f1']:.4f}\n")
                f.write(f"  Combined F1: {row['combined_f1']:.4f}\n")
                f.write("-" * 40 + "\n")
            
            # Best model summary
            best_model = df.iloc[0]
            f.write(f"\nBest Overall Model: {best_model['model_name']}\n")
            f.write(f"Combined F1 Score: {best_model['combined_f1']:.4f}\n")
        
        logger.info(f"Model comparison saved to {csv_path}")
        logger.info(f"Training summary saved to {summary_path}")
        
        # Print summary to console
        self._print_summary(df)
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary to console"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        
        for _, row in df.iterrows():
            logger.info(f"\n{row['model_name']}:")
            logger.info(f"  Age: Acc={row['age_accuracy']:.4f}, F1={row['age_f1']:.4f}")
            logger.info(f"  Gender: Acc={row['gender_accuracy']:.4f}, F1={row['gender_f1']:.4f}")
            logger.info(f"  Training Time: {row['training_time_seconds']:.1f}s")
        
        best_model = df.iloc[0]
        logger.info(f"\n🏆 Best Model: {best_model['model_name']}")
        logger.info(f"   Combined F1: {best_model['combined_f1']:.4f}")
        logger.info("="*60)
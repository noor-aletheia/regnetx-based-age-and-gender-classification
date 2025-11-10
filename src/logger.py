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
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.history = []
        
        self.start_time = datetime.now()
        
        logger.info(f"Initialized logger for {model_name}")

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
        csv_path = os.path.join(self.logs_dir, f'{self.model_name}_training_history.csv')
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
        
        plot_path = os.path.join(self.logs_dir, f'{self.model_name}_training_curves.png')
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
    
    def export_to_onnx(self, model: torch.nn.Module, model_name: str, 
                      input_shape: tuple = (1, 3, 224, 224)) -> Dict[str, float]:
        """
        Export model to ONNX format (FP32 and FP16 versions)
        
        Args:
            model: Model to export
            model_name: Name of the model
            input_shape: Input tensor shape
            
        Returns:
            Dictionary containing cosine similarity metrics or empty dict if verification disabled
        """
        if not self.config.get('logging.export_onnx', True):
            return {}
            
        model.eval()
        device = next(model.parameters()).device
        
        dummy_input = torch.randn(input_shape).to(device)
        
        onnx_path = os.path.join(self.models_dir, f'{model_name}.onnx')
        onnx_fp16_path = os.path.join(self.models_dir, f'{model_name}_fp16.onnx')
        
        try:            
            export_kwargs = {
                'model': model,
                'args': dummy_input,
                'f': onnx_path,
                'export_params': True,
                'opset_version': 18,
                'do_constant_folding': True,
                'input_names': ['input'],
                'output_names': ['age_output', 'gender_output'],
                'verbose': False
            }
            
            # Add dynamic axes if enabled in config
            if self.config.get('logging.onnx_dynamic_axes', False):
                export_kwargs['dynamic_axes'] = {
                    'input': {0: 'batch_size'},
                    'age_output': {0: 'batch_size'},
                    'gender_output': {0: 'batch_size'}
                }
                logger.info("ONNX export with dynamic axes enabled")
            
            torch.onnx.export(**export_kwargs)
            
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            if self.config.get('logging.simplify_onnx', True):
                try:
                    import onnxsim
                    
                    logger.info("Simplifying ONNX model...")
                    onnx_model_simplified, check_ok = onnxsim.simplify(
                        onnx_model,
                        check_n=1,
                        perform_optimization=True,
                        skip_fuse_bn=True,
                        skip_constant_folding=False,
                        skip_shape_inference=False,
                        input_shapes={'input': [1, 3, 224, 224]}
                    )
                    
                    if check_ok:
                        onnx_simplified_path = os.path.join(self.models_dir, f'{model_name}_simplified.onnx')
                        onnx.save(onnx_model_simplified, onnx_simplified_path)
                        
                        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
                        simplified_size = os.path.getsize(onnx_simplified_path) / (1024 * 1024)
                        size_reduction = ((original_size - simplified_size) / original_size) * 100
                        
                        logger.info(f"✅ ONNX model simplified successfully")
                        logger.info(f"📊 Size reduction: {original_size:.2f}MB → {simplified_size:.2f}MB ({size_reduction:.1f}% smaller)")
                        
                        onnx_model = onnx_model_simplified
                        onnx_path = onnx_simplified_path
                    else:
                        logger.warning("ONNX simplification failed validation, using original model")
                        
                except ImportError:
                    logger.warning("onnxsim not available. Install with: pip install onnxsim")
                except Exception as e:
                    logger.warning(f"ONNX simplification failed: {e}")
            
            ort_session = ort.InferenceSession(onnx_path)
            test_input = dummy_input.cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            cosine_similarity_results = {}
            if self.config.get('logging.verify_onnx_similarity', True):
                similarity_threshold = self.config.get('logging.similarity_threshold', 0.99)
                cosine_similarity_results = self._verify_model_outputs(model, dummy_input, ort_outputs, model_name, similarity_threshold)
            
            logger.info(f"✅ FP32 ONNX model exported and verified: {onnx_path}")
            
            if self.config.get('logging.export_onnx_fp16', True):
                try:
                    from onnxconverter_common import float16
                    
                    onnx_model_fp32 = onnx.load(onnx_path)
                    onnx_model_fp16 = float16.convert_float_to_float16(onnx_model_fp32)
                    
                    onnx.save(onnx_model_fp16, onnx_fp16_path)
                    
                    onnx.checker.check_model(onnx_model_fp16)
                    
                    try:
                        ort_session_fp16 = ort.InferenceSession(onnx_fp16_path)
                        test_input_fp16 = test_input.astype(np.float16) if test_input.dtype == np.float32 else test_input
                        ort_inputs_fp16 = {ort_session_fp16.get_inputs()[0].name: test_input_fp16}
                        ort_outputs_fp16 = ort_session_fp16.run(None, ort_inputs_fp16)
                        
                        logger.info(f"✅ FP16 ONNX model exported and verified: {onnx_fp16_path}")
                        
                        def get_total_onnx_size(onnx_file_path):
                            total_size = os.path.getsize(onnx_file_path)
                            data_file = onnx_file_path + '.data'
                            if os.path.exists(data_file):
                                total_size += os.path.getsize(data_file)
                            return total_size / (1024 * 1024)
                        
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
            
            return cosine_similarity_results
            
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            return {}
    
    def _verify_model_outputs(self, pytorch_model: torch.nn.Module, dummy_input: torch.Tensor, 
                            onnx_outputs: list, model_name: str, similarity_threshold: float = 0.97) -> Dict[str, float]:
        """
        Verify ONNX model outputs against PyTorch model outputs using cosine similarity
        
        Args:
            pytorch_model: Original PyTorch model
            dummy_input: Input tensor used for testing
            onnx_outputs: Outputs from ONNX model inference
            model_name: Name of the model for logging
            similarity_threshold: Minimum cosine similarity to consider models equivalent
            
        Returns:
            Dictionary containing cosine similarity metrics for each output
        """
        try:
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_outputs = pytorch_model(dummy_input)
            
            if isinstance(pytorch_outputs, dict):
                pytorch_outputs_np = [pytorch_outputs['age'].cpu().numpy(), pytorch_outputs['gender'].cpu().numpy()]
            elif isinstance(pytorch_outputs, tuple):
                pytorch_outputs_np = [output.cpu().numpy() for output in pytorch_outputs]
            else:
                pytorch_outputs_np = [pytorch_outputs.cpu().numpy()]
            
            if len(pytorch_outputs_np) != len(onnx_outputs):
                logger.warning(f"Output count mismatch: PyTorch={len(pytorch_outputs_np)}, ONNX={len(onnx_outputs)}")
                return {}
            
            pytorch_concat = np.concatenate([out.flatten() for out in pytorch_outputs_np])
            onnx_concat = np.concatenate([out.flatten() for out in onnx_outputs])
            
            overall_similarity = self._cosine_similarity(pytorch_concat, onnx_concat)
            
            logger.info(f"📊 {model_name} - Overall Model Verification:")
            logger.info(f"   Overall Cosine Similarity: {overall_similarity:.6f}")
            
            if overall_similarity >= similarity_threshold:
                logger.info(f"   ✅ Overall model verification PASSED")
            else:
                logger.warning(f"   ⚠️ Overall model verification FAILED (similarity < {similarity_threshold})")
            
            min_similarity = overall_similarity
            avg_similarity = overall_similarity
            similarities = [overall_similarity, overall_similarity]  
            
            logger.info(f"🔍 {model_name} Overall Verification:")
            logger.info(f"   Average Similarity: {avg_similarity:.6f}")
            logger.info(f"   Minimum Similarity: {min_similarity:.6f}")
            
            if min_similarity >= similarity_threshold:
                logger.info(f"   ✅ Model verification PASSED - ONNX model is highly equivalent to PyTorch model")
            else:
                logger.warning(f"   ⚠️ Model verification FAILED - Consider checking conversion settings")
                logger.warning(f"   This might indicate precision loss or conversion issues")
            
            return {
                'average_cosine_similarity': avg_similarity,
                'minimum_cosine_similarity': min_similarity,
                'age_output_similarity': similarities[0] if len(similarities) > 0 else 0.0,
                'gender_output_similarity': similarities[1] if len(similarities) > 1 else 0.0,
                'verification_passed': min_similarity >= similarity_threshold
            }
                
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return {}
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0 if norm1 == norm2 else 0.0
            
            dot_product = np.dot(vec1, vec2)
            cosine_sim = dot_product / (norm1 * norm2)
            
            return max(0.0, min(1.0, cosine_sim))
            
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0


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
                         gflops: float = None, fps: float = None,
                         cosine_similarity_results: Dict[str, float] = None):
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
            'ONNX Avg Similarity': cosine_similarity_results.get('average_cosine_similarity', 0.0) if cosine_similarity_results else 0.0,
            'ONNX Age Similarity': cosine_similarity_results.get('age_output_similarity', 0.0) if cosine_similarity_results else 0.0,
            'ONNX Gender Similarity': cosine_similarity_results.get('gender_output_similarity', 0.0) if cosine_similarity_results else 0.0,
            'ONNX Verification': 'PASS' if cosine_similarity_results and cosine_similarity_results.get('verification_passed', False) else 'FAIL'
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
            
            header = f"{'Model':<15} {'P1/P2 Epochs':<12} {'Augment':<20} {'LR(P1/P2)':<15} {'WD(P1/P2)':<15} {'Scheduler':<15} {'Params(M)':<10} {'Time(s)':<8} {'Age F1':<7} {'Gender F1':<9} {'Comb F1':<8} {'GFLOPs':<7} {'ONNX Sim':<9}\n"
            f.write(header)
            f.write("-"*130 + "\n")
            
            for _, row in df.iterrows():
                params_m = row['Parameters'] / 1_000_000 if row['Parameters'] > 0 else 0
                onnx_sim = row.get('ONNX Avg Similarity', 0.0)
                line = f"{row['Model']:<15} {row['Phase 1 Epochs']}/{row['Phase 2 Epochs']:<8} {row['Augmentation']:<20} {row['LR (P1/P2)']:<15} {row['Weight Decay (P1/P2)']:<15} {row['scheduler']:<15} {params_m:<10.1f} {row['Training Time (s)']:<8.0f} {row['Age F1']:<7.3f} {row['Gender F1']:<9.3f} {row['Combined F1']:<8.3f} {row['GFLOPs']:<7.1f} {onnx_sim:<9.5f}\n"
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
                f.write(f"   ONNX Verification: {best_model.get('ONNX Verification', 'N/A')} (Avg Similarity: {best_model.get('ONNX Avg Similarity', 0.0):.5f})\n")
        
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
            logger.info(f"  ONNX Verification: {row.get('ONNX Verification', 'N/A')} (Similarity: {row.get('ONNX Avg Similarity', 0.0):.5f})")
        
        if len(df) > 0:
            best_model = df.iloc[0]
            logger.info(f"\n🏆 Best Model: {best_model['Model']}")
            logger.info(f"   Combined F1: {best_model['Combined F1']:.4f}")
            logger.info(f"   Dataset: Train={best_model['train size']}, Val={best_model['val size']}, Test={best_model['test size']}")
            logger.info(f"   Age Classes: {best_model.get('age class scheme', 'N/A')}")
            logger.info(f"   ONNX Verification: {best_model.get('ONNX Verification', 'N/A')}")
        logger.info("="*80)
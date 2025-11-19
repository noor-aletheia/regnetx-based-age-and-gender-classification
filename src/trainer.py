"""
Training logic with two-phase training and automatic phase transition
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
import os
from config import Config
from models import DualHeadRegNetX, create_loss_functions, create_optimizer, create_scheduler

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        val_loss = float(val_loss)
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            
        return self.should_stop


class MetricsCalculator:
    """Calculate and track training metrics including confusion matrices"""
    
    def __init__(self, class_names: Dict[str, List[str]]):
        """
        Initialize metrics calculator
        
        Args:
            class_names: Dictionary containing class names for age and gender
        """
        self.class_names = class_names
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task: str) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            task: Task name ('age' or 'gender')
            
        Returns:
            Dictionary containing accuracy, precision, recall, F1 scores, and confusion matrix
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
        }
        
        class_names = self.class_names[task]
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def save_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            task: str, save_path: str, title: str = None):
        """
        Save confusion matrix as a heatmap
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            task: Task name ('age' or 'gender')
            save_path: Path to save the plot
            title: Optional title for the plot
        """
        plt.figure(figsize=(10, 8))
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Frequency'})
        
        plt.title(title or f'{task.capitalize()} Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {save_path}")
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str], task: str):
        """
        Print detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            task: Task name ('age' or 'gender')
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     zero_division=0)
        logger.info(f"\n{task.capitalize()} Classification Report:\n{report}")
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, task: str) -> np.ndarray:
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred)


class TwoPhaseTrainer:
    """Two-phase trainer for age and gender classification"""
    
    def __init__(self, model: DualHeadRegNetX, config: Config, 
                 age_class_weights: torch.Tensor = None,
                 gender_class_weights: torch.Tensor = None):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            config: Configuration object
            age_class_weights: Optional class weights for age classification
            gender_class_weights: Optional class weights for gender classification
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.get('hardware.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.mixed_precision = config.get('hardware.mixed_precision', True)
        
        self.model.to(self.device)
        
        self.age_loss_fn, self.gender_loss_fn = create_loss_functions(
            config, self.device, age_class_weights, gender_class_weights
        )
        
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        self.scaler = GradScaler() if self.mixed_precision else None
        
        self.metrics_calculator = None
        
        self.current_phase = 1
        self.epoch = 0
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_age_accuracy': 0.0,
            'val_gender_accuracy': 0.0,
            'val_age_f1': 0.0,
            'val_gender_f1': 0.0
        }
        
        self.early_stopping = EarlyStopping(
            patience=int(config.get('training.phase1.patience', 3)),
            min_delta=0.001
        )
        
        self.age_loss_weight = float(config.get('training.age_loss_weight', 1.0))
        self.gender_loss_weight = float(config.get('training.gender_loss_weight', 1.0))
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
    
    def initialize_metrics_calculator(self, age_class_names: List[str], gender_class_names: List[str]):
        """
        Initialize metrics calculator with actual class names from dataset
        
        Args:
            age_class_names: List of age class names
            gender_class_names: List of gender class names
        """
        class_names = {
            'age': age_class_names,
            'gender': gender_class_names
        }
        self.metrics_calculator = MetricsCalculator(class_names)
        logger.info(f"Metrics calculator initialized with {len(age_class_names)} age classes and {len(gender_class_names)} gender classes")
    
    def _transition_to_phase2(self):
        """Transition from Phase 1 to Phase 2"""
        logger.info("=== Transitioning to Phase 2 (Full Fine-tuning) ===")
        
        self.model.unfreeze_backbone()
        
        self.current_phase = 2
        
        self.optimizer = torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=float(self.config.get('training.phase2.learning_rate', 1e-4)),
            weight_decay=float(self.config.get('training.phase2.weight_decay', 1e-5))
        )
        
        self.scheduler = create_scheduler(self.optimizer, self.config)
        
        self.early_stopping = EarlyStopping(
            patience=int(self.config.get('training.phase2.patience', 5)),
            min_delta=0.001
        )
        
        logger.info("Phase 2 transition completed")
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform forward pass and calculate loss
        
        Args:
            batch: Batch of data
            
        Returns:
            Tuple of (outputs, total_loss)
        """
        images = batch['image'].to(self.device)
        age_labels = batch['age'].to(self.device)
        gender_labels = batch['gender'].to(self.device)
        
        if self.mixed_precision:
            with autocast():
                outputs = self.model(images)
                age_loss = self.age_loss_fn(outputs['age'], age_labels)
                gender_loss = self.gender_loss_fn(outputs['gender'], gender_labels)
                total_loss = (self.age_loss_weight * age_loss + 
                             self.gender_loss_weight * gender_loss)
        else:
            outputs = self.model(images)
            age_loss = self.age_loss_fn(outputs['age'], age_labels)
            gender_loss = self.gender_loss_fn(outputs['gender'], gender_labels)
            total_loss = (self.age_loss_weight * age_loss + 
                         self.gender_loss_weight * gender_loss)
        
        return outputs, total_loss, age_loss, gender_loss
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_age_loss = 0.0
        running_gender_loss = 0.0
        num_batches = 0
        
        age_predictions = []
        gender_predictions = []
        age_targets = []
        gender_targets = []
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            outputs, total_loss, age_loss, gender_loss = self._forward_pass(batch)
            
            if self.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            running_loss += total_loss.item()
            running_age_loss += age_loss.item()
            running_gender_loss += gender_loss.item()
            num_batches += 1
            
            age_pred = torch.argmax(outputs['age'], dim=1).cpu().numpy()
            gender_pred = torch.argmax(outputs['gender'], dim=1).cpu().numpy()
            
            age_predictions.extend(age_pred)
            gender_predictions.extend(gender_pred)
            age_targets.extend(batch['age'].cpu().numpy())
            gender_targets.extend(batch['gender'].cpu().numpy())
        
        age_metrics = self.metrics_calculator.calculate_metrics(
            np.array(age_targets), np.array(age_predictions), 'age'
        )
        gender_metrics = self.metrics_calculator.calculate_metrics(
            np.array(gender_targets), np.array(gender_predictions), 'gender'
        )
        
        return {
            'train_loss': running_loss / num_batches,
            'train_age_loss': running_age_loss / num_batches,
            'train_gender_loss': running_gender_loss / num_batches,
            'train_age_accuracy': age_metrics['accuracy'],
            'train_age_f1': age_metrics['f1'],
            'train_gender_accuracy': gender_metrics['accuracy'],
            'train_gender_f1': gender_metrics['f1']
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_age_loss = 0.0
        running_gender_loss = 0.0
        num_batches = 0
        
        age_predictions = []
        gender_predictions = []
        age_targets = []
        gender_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs, total_loss, age_loss, gender_loss = self._forward_pass(batch)
                
                running_loss += total_loss.item()
                running_age_loss += age_loss.item()
                running_gender_loss += gender_loss.item()
                num_batches += 1
                
                age_pred = torch.argmax(outputs['age'], dim=1).cpu().numpy()
                gender_pred = torch.argmax(outputs['gender'], dim=1).cpu().numpy()
                
                age_predictions.extend(age_pred)
                gender_predictions.extend(gender_pred)
                age_targets.extend(batch['age'].cpu().numpy())
                gender_targets.extend(batch['gender'].cpu().numpy())
        
        age_metrics = self.metrics_calculator.calculate_metrics(
            np.array(age_targets), np.array(age_predictions), 'age'
        )
        gender_metrics = self.metrics_calculator.calculate_metrics(
            np.array(gender_targets), np.array(gender_predictions), 'gender'
        )
        
        return {
            'val_loss': running_loss / num_batches,
            'val_age_loss': running_age_loss / num_batches,
            'val_gender_loss': running_gender_loss / num_batches,
            'val_age_accuracy': age_metrics['accuracy'],
            'val_age_precision': age_metrics['precision'],
            'val_age_recall': age_metrics['recall'],
            'val_age_f1': age_metrics['f1'],
            'val_gender_accuracy': gender_metrics['accuracy'],
            'val_gender_precision': gender_metrics['precision'],
            'val_gender_recall': gender_metrics['recall'],
            'val_gender_f1': gender_metrics['f1']
        }
    
    def _should_transition_to_phase2(self, val_metrics: Dict[str, float]) -> bool:
        """Check if we should transition to Phase 2"""
        if self.current_phase != 1:
            return False
            

        max_phase1_epochs = int(self.config.get('training.phase1.epochs', 10))
        return (self.early_stopping.counter >= self.early_stopping.patience or 
                self.epoch >= max_phase1_epochs)
    
    def _update_best_metrics(self, val_metrics: Dict[str, float]):
        """Update best metrics if current ones are better"""
        current_val_loss = float(val_metrics['val_loss'])
        best_val_loss = float(self.best_metrics['val_loss'])
        if current_val_loss < best_val_loss:
            self.best_metrics.update(val_metrics)
            return True
        return False
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the model with two-phase training
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.model.model_name}")
        logger.info("=== Phase 1: Frozen Backbone Training ===")
        
        self.model.freeze_backbone()
        
        history = {
            'epoch': [],
            'phase': [],
            'train_loss': [],
            'val_loss': [],
            'train_age_accuracy': [],
            'val_age_accuracy': [],
            'train_gender_accuracy': [],
            'val_gender_accuracy': [],
            'train_age_f1': [],
            'val_age_f1': [],
            'train_gender_f1': [],
            'val_gender_f1': [],
            'learning_rate': []
        }
        
        start_time = time.time()
        max_epochs = (int(self.config.get('training.phase1.epochs', 10)) + 
                     int(self.config.get('training.phase2.epochs', 15)))
        
        for epoch in range(max_epochs):
            self.epoch = epoch
            
            train_metrics = self._train_epoch(train_loader)
            
            val_metrics = self._validate_epoch(val_loader)
            
            epoch_metrics = {**train_metrics, **val_metrics}
            
            history['epoch'].append(epoch)
            history['phase'].append(self.current_phase)
            for key, value in epoch_metrics.items():
                if key in history:
                    history[key].append(value)
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            self._log_epoch_progress(epoch, epoch_metrics)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            if self._should_transition_to_phase2(val_metrics):
                if self.current_phase == 1:
                    self._transition_to_phase2()
                    self.early_stopping = EarlyStopping(
                        patience=int(self.config.get('training.phase2.patience', 5)),
                        min_delta=0.001
                    )
                    continue
            
            if self.early_stopping(val_metrics['val_loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            if self._update_best_metrics(val_metrics):
                logger.info("New best model found!")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return history
    
    def _log_epoch_progress(self, epoch: int, metrics: Dict[str, float]):
        """Log progress for current epoch"""
        logger.info(
            f"Phase {self.current_phase} | Epoch {epoch:3d} | "
            f"Train Loss: {metrics['train_loss']:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Age Acc: {metrics['val_age_accuracy']:.4f} | "
            f"Gender Acc: {metrics['val_gender_accuracy']:.4f} | "
            f"Age F1: {metrics['val_age_f1']:.4f} | "
            f"Gender F1: {metrics['val_gender_f1']:.4f} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
    
    def evaluate(self, test_loader: DataLoader, save_confusion_matrix: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on test set with comprehensive metrics
        
        Args:
            test_loader: Test data loader
            save_confusion_matrix: Whether to save confusion matrix plots
            
        Returns:
            Test metrics and predictions
        """
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        
        age_predictions = []
        gender_predictions = []
        age_targets = []
        gender_targets = []

        # --- FPS calculation: start timing ---
        inference_times = []
        total_start_time = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                batch_start_time = time.time()
                outputs, _, _, _ = self._forward_pass(batch)
                batch_time = time.time() - batch_start_time
                inference_times.append(batch_time)

                age_pred = torch.argmax(outputs['age'], dim=1).cpu().numpy()
                gender_pred = torch.argmax(outputs['gender'], dim=1).cpu().numpy()

                age_predictions.extend(age_pred)
                gender_predictions.extend(gender_pred)
                age_targets.extend(batch['age'].cpu().numpy())
                gender_targets.extend(batch['gender'].cpu().numpy())

        total_time = time.time() - total_start_time
        
        age_targets = np.array(age_targets)
        age_predictions = np.array(age_predictions)
        gender_targets = np.array(gender_targets)
        gender_predictions = np.array(gender_predictions)
        
        age_metrics = self.metrics_calculator.calculate_metrics(
            age_targets, age_predictions, 'age'
        )
        gender_metrics = self.metrics_calculator.calculate_metrics(
            gender_targets, gender_predictions, 'gender'
        )
        
        age_class_names = self.metrics_calculator.class_names['age']
        gender_class_names = self.metrics_calculator.class_names['gender']
        
        self.metrics_calculator.print_classification_report(
            age_targets, age_predictions, age_class_names, 'age'
        )
        self.metrics_calculator.print_classification_report(
            gender_targets, gender_predictions, gender_class_names, 'gender'
        )
        if save_confusion_matrix:
            # Try to use the same directory as training curves/history (logger directory)
            # Fallback to output.logs_dir/model_name if not available
            log_dir = getattr(self, 'logger_dir', None)
            if log_dir is None:
                output_dir = self.config.get('output.logs_dir', './outputs/logs')
                model_name = getattr(self.model, 'model_name', None)
                if not model_name:
                    model_name = self.config.get('experiment.model_name', None)
                if not model_name:
                    model_name = self.config.get('experiment_name', 'model')
                log_dir = os.path.join(output_dir, str(model_name))
            os.makedirs(log_dir, exist_ok=True)
            model_name = getattr(self.model, 'model_name', None)
            if not model_name:
                model_name = self.config.get('experiment.model_name', None)
            if not model_name:
                model_name = self.config.get('experiment_name', 'model')
            age_cm_path = os.path.join(log_dir, f'confusion_matrix_age_{model_name}.png')
            self.metrics_calculator.save_confusion_matrix(
                age_metrics['confusion_matrix'], 
                age_class_names, 
                'age', 
                age_cm_path,
                f'Age Classification Confusion Matrix (Acc: {age_metrics["accuracy"]:.3f})'
            )
            gender_cm_path = os.path.join(log_dir, f'confusion_matrix_gender_{model_name}.png')
            self.metrics_calculator.save_confusion_matrix(
                gender_metrics['confusion_matrix'], 
                gender_class_names, 
                'gender', 
                gender_cm_path,
                f'Gender Classification Confusion Matrix (Acc: {gender_metrics["accuracy"]:.3f})'
            )
        
        # --- FPS calculation ---
        total_samples = len(age_predictions)
        avg_batch_time = np.mean(inference_times) if inference_times else 0.0
        fps = test_loader.batch_size / avg_batch_time if avg_batch_time > 0 else 0.0
        overall_fps = total_samples / total_time if total_time > 0 else 0.0

        test_results = {
            'age_metrics': age_metrics,
            'gender_metrics': gender_metrics,
            'age_confusion_matrix': age_metrics['confusion_matrix'],
            'gender_confusion_matrix': gender_metrics['confusion_matrix'],
            'age_predictions': age_predictions.tolist(),
            'gender_predictions': gender_predictions.tolist(),
            'age_targets': age_targets.tolist(),
            'gender_targets': gender_targets.tolist(),
            'performance': {
                'total_inference_time': total_time,
                'avg_batch_time': avg_batch_time,
                'fps_per_batch': fps,
                'overall_fps': overall_fps,
                'samples_per_second': overall_fps
            }
        }

        logger.info("=== Test Results ===")
        logger.info(f"Age - Accuracy: {age_metrics['accuracy']:.4f}, Precision: {age_metrics['precision']:.4f}, Recall: {age_metrics['recall']:.4f}, F1: {age_metrics['f1']:.4f}")
        logger.info(f"Gender - Accuracy: {gender_metrics['accuracy']:.4f}, Precision: {gender_metrics['precision']:.4f}, Recall: {gender_metrics['recall']:.4f}, F1: {gender_metrics['f1']:.4f}")
        logger.info(f"Test set FPS: {overall_fps:.2f} samples/second, Total inference time: {total_time:.2f}s, Avg batch time: {avg_batch_time:.4f}s")
        
        return test_results
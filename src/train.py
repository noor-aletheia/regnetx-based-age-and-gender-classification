"""
Main training script for RegNetX age and gender classification
"""
import os
import sys
import torch
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Add src to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import DataProcessor
from models import ModelFactory
from trainer import TwoPhaseTrainer
from logger import TrainingLogger, ModelSaver, ResultsAggregator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/outputs/logs/training.log')
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Setup training environment"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    return device

def train_single_model(model_name: str, config: Config, data_processor: DataProcessor,
                      results_aggregator: ResultsAggregator) -> Dict[str, Any]:
    """
    Train a single model variant
    
    Args:
        model_name: Name of the model variant
        config: Configuration object
        data_processor: Data processor instance
        results_aggregator: Results aggregator instance
        
    Returns:
        Training results dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODEL: {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    # Create model
    model = ModelFactory.create_model(config, model_name)
    ModelFactory.print_model_summary(model)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_processor.get_dataloaders()
    
    # Get class weights for handling imbalance
    age_class_weights, gender_class_weights = data_processor.get_class_weights()
    
    # Initialize trainer
    trainer = TwoPhaseTrainer(
        model=model,
        config=config,
        age_class_weights=age_class_weights,
        gender_class_weights=gender_class_weights
    )
    
    # Initialize metrics calculator with actual class names
    age_class_names, gender_class_names = data_processor.get_class_names()
    trainer.initialize_metrics_calculator(age_class_names, gender_class_names)
    
    # Initialize logger and saver
    experiment_name = config.get('experiment_name', None)
    training_logger = TrainingLogger(config, model_name, experiment_name=experiment_name)
    model_saver = ModelSaver(config)
    
    try:
        # Train model
        logger.info("Starting training...")
        import traceback
        try:
            history = trainer.train(train_loader, val_loader)
        except Exception as e:
            logger.error(f"Detailed error during training: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Log training history
        for i, epoch_data in enumerate(history['epoch']):
            epoch_metrics = {}
            for key in history:
                if key != 'epoch' and i < len(history[key]):
                    epoch_metrics[key] = history[key][i]
            
            training_logger.log_epoch(
                epoch=epoch_data,
                metrics=epoch_metrics,
                phase=history['phase'][i]
            )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_loader)
        
        # Log test results
        training_logger.log_test_results(test_results)
        
        # Save best model
        best_metrics = {
            'val_loss': trainer.best_metrics['val_loss'],
            'val_age_accuracy': trainer.best_metrics['val_age_accuracy'],
            'val_gender_accuracy': trainer.best_metrics['val_gender_accuracy'],
            'val_age_f1': trainer.best_metrics['val_age_f1'],
            'val_gender_f1': trainer.best_metrics['val_gender_f1'],
            'test_age_accuracy': test_results['age_metrics']['accuracy'],
            'test_gender_accuracy': test_results['gender_metrics']['accuracy'],
            'test_age_f1': test_results['age_metrics']['f1'],
            'test_gender_f1': test_results['gender_metrics']['f1']
        }
        
        model_path = model_saver.save_best_model(model, best_metrics, model_name)
        
        # Export to ONNX
        if config.get('logging.export_onnx', True):
            model_saver.export_to_onnx(model, model_name)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Add to results aggregator
        model_info = ModelFactory.get_model_info(model)
        results_aggregator.add_model_results(
            model_name=model_name,
            test_results=test_results,
            training_time=training_time,
            model_info=model_info
        )
        
        # Close logger
        training_logger.close()
        
        logger.info(f"✅ Model {model_name} training completed successfully!")
        logger.info(f"   Training time: {training_time:.1f}s")
        logger.info(f"   Test Age Accuracy: {test_results['age_metrics']['accuracy']:.4f}")
        logger.info(f"   Test Gender Accuracy: {test_results['gender_metrics']['accuracy']:.4f}")
        
        return {
            'model_name': model_name,
            'success': True,
            'training_time': training_time,
            'test_results': test_results,
            'model_path': model_path,
            'best_metrics': best_metrics
        }
        
    except Exception as e:
        logger.error(f"❌ Error training model {model_name}: {str(e)}")
        training_logger.close()
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }

def main():
    """Main training function"""
    logger.info("Starting RegNetX Age & Gender Classification Training")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup environment
        device = setup_environment()
        # Load configuration
        config = Config('config.yaml')
        config.create_directories()
        # Log configuration
        logger.info("Configuration loaded:")
        logger.info(f"  Dataset path: {config.get('dataset.path')}")
        logger.info(f"  Batch size: {config.get('dataset.batch_size')}")
        logger.info(f"  Models to train: {config.get('models.variants')}")
        logger.info(f"  Mixed precision: {config.get('hardware.mixed_precision')}")
        # Initialize data processor
        logger.info("Loading and processing dataset...")
        data_processor = DataProcessor(config)
        # Initialize results aggregator
        results_aggregator = ResultsAggregator(config)
        # Get model variants to train
        model_variants = config.get('models.variants', ['mobilenetv3_small_075'])
        # Train each model sequentially
        all_results = []
        successful_models = 0
        total_training_time = 0
        for model_name in model_variants:
            result = train_single_model(
                model_name=model_name,
                config=config,
                data_processor=data_processor,
                results_aggregator=results_aggregator
            )
            all_results.append(result)
            if result['success']:
                successful_models += 1
                total_training_time += result['training_time']
            else:
                logger.error(f"Failed to train {model_name}: {result.get('error', 'Unknown error')}")
        # Save comparison results
        if successful_models > 0:
            results_aggregator.save_comparison_table()
        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Models trained successfully: {successful_models}/{len(model_variants)}")
        logger.info(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
        if successful_models == 0:
            logger.error("❌ No models were trained successfully!")
            sys.exit(1)
        else:
            logger.info("✅ Training pipeline completed successfully!")
            # Show best model
            if results_aggregator.results:
                best_result = max(results_aggregator.results, 
                                key=lambda x: x['age_f1'] + x['gender_f1'])
                logger.info(f"🏆 Best model: {best_result['model_name']}")
                logger.info(f"   Combined F1: {best_result['age_f1'] + best_result['gender_f1']:.4f}")
        logger.info(f"\n📊 Results saved to: {config.get('output.logs_dir')}")
        logger.info(f"💾 Models saved to: {config.get('output.models_dir')}")
    except Exception as e:
        logger.error(f"❌ Fatal error in training pipeline: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
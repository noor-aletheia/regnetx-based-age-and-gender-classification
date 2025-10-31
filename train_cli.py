"""
Enhanced CLI training script for RegNetX age and gender classification
Supports individual model training, runtime config changes, and optional data augmentation
"""
import os
import sys
import torch
import logging
import argparse
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import Config
from dataset import DataProcessor
from models import ModelFactory
from trainer import TwoPhaseTrainer
from logger import TrainingLogger, ModelSaver, ResultsAggregator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='RegNetX Age & Gender Classification Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models (default)
  python train_cli.py

  # Train specific model
  python train_cli.py --model regnetx_008

  # Train multiple specific models
  python train_cli.py --model regnetx_006 regnetx_016

  # List available models
  python train_cli.py --list-models

  # Train with custom parameters
  python train_cli.py --model regnetx_008 --batch-size 32 --lr 0.001 --epochs 20

  # Train without data augmentation
  python train_cli.py --model regnetx_008 --no-augmentation

  # Train with custom input size
  python train_cli.py --model regnetx_016 --input-size 448
        """
    )
    
    # Model selection
    parser.add_argument('--model', '--models', nargs='*', 
                       help='Model(s) to train. If not specified, trains all models from config.')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    
    # Configuration
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    
    # Dataset parameters
    parser.add_argument('--data-path', 
                       help='Path to dataset (overrides config)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (overrides config)')
    parser.add_argument('--input-size', type=int,
                       help='Input image size (overrides config)')
    parser.add_argument('--num-workers', type=int,
                       help='Number of data loading workers (overrides config)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int,
                       help='Total number of epochs (overrides both phases)')
    parser.add_argument('--phase1-epochs', type=int,
                       help='Phase 1 epochs (frozen backbone)')
    parser.add_argument('--phase2-epochs', type=int,
                       help='Phase 2 epochs (full fine-tuning)')
    parser.add_argument('--lr', '--learning-rate', type=float,
                       help='Learning rate for Phase 1 (overrides config)')
    parser.add_argument('--lr2', '--learning-rate-phase2', type=float,
                       help='Learning rate for Phase 2 (overrides config)')
    parser.add_argument('--weight-decay', type=float,
                       help='Weight decay (overrides config)')
    
    # Augmentation
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--augmentation-strength', choices=['light', 'medium', 'strong'],
                       help='Augmentation strength preset')
    
    # Hardware
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                       default='auto', help='Device to use for training')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    
    # Output
    parser.add_argument('--output-dir', 
                       help='Output directory (overrides config)')
    parser.add_argument('--experiment-name',
                       help='Experiment name (added to output paths)')
    parser.add_argument('--save-every', type=int,
                       help='Save checkpoint every N epochs')
    
    # Debugging
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration and exit without training')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    # Class weight balancing (power scaling)
    parser.add_argument('--class_weight_mode', choices=['none', 'auto', 'power', 'manual'],
                        help='Class weight calculation mode: none, auto, power, or manual (overrides config)')
    parser.add_argument('--class_weight_power_alpha', type=float,
                        help='Alpha value for power scaling of class weights (overrides config)')
    
    return parser.parse_args()

def create_runtime_config(args, base_config_path: str) -> Config:
    """Create runtime configuration by merging CLI args with base config"""
    
    # Load base configuration
    config = Config(base_config_path)
    
    # Override dataset parameters
    if args.data_path:
        config.set('dataset.path', args.data_path)
    if args.batch_size:
        config.set('dataset.batch_size', args.batch_size)
    if args.input_size:
        config.set('dataset.image_size', args.input_size)
    if args.num_workers:
        config.set('dataset.num_workers', args.num_workers)
    
    # Override training parameters
    if args.epochs:
        # Split total epochs between phases if not specified separately
        if not args.phase1_epochs and not args.phase2_epochs:
            phase1_epochs = max(5, args.epochs // 3)
            phase2_epochs = args.epochs - phase1_epochs
            config.set('training.phase1.epochs', phase1_epochs)
            config.set('training.phase2.epochs', phase2_epochs)
        else:
            config.set('training.phase1.epochs', args.epochs)
            config.set('training.phase2.epochs', args.epochs)
    
    if args.phase1_epochs:
        config.set('training.phase1.epochs', args.phase1_epochs)
    if args.phase2_epochs:
        config.set('training.phase2.epochs', args.phase2_epochs)
    if args.lr:
        config.set('training.phase1.learning_rate', args.lr)
    if args.lr2:
        config.set('training.phase2.learning_rate', args.lr2)
    if args.weight_decay:
        config.set('training.phase1.weight_decay', args.weight_decay)
        config.set('training.phase2.weight_decay', args.weight_decay)
    
    # Handle augmentation settings
    if args.no_augmentation:
        config.set('augmentation.train.horizontal_flip', 0.0)
        config.set('augmentation.train.rotation', 0)
        config.set('augmentation.train.color_jitter.brightness', 0.0)
        config.set('augmentation.train.color_jitter.contrast', 0.0)
        config.set('augmentation.train.color_jitter.saturation', 0.0)
        config.set('augmentation.train.color_jitter.hue', 0.0)
        logger.info("Data augmentation disabled")
    
    elif args.augmentation_strength:
        if args.augmentation_strength == 'light':
            config.set('augmentation.train.horizontal_flip', 0.3)
            config.set('augmentation.train.rotation', 3)
            config.set('augmentation.train.color_jitter.brightness', 0.05)
            config.set('augmentation.train.color_jitter.contrast', 0.05)
        elif args.augmentation_strength == 'medium':
            config.set('augmentation.train.horizontal_flip', 0.5)
            config.set('augmentation.train.rotation', 5)
            config.set('augmentation.train.color_jitter.brightness', 0.1)
            config.set('augmentation.train.color_jitter.contrast', 0.1)
        elif args.augmentation_strength == 'strong':
            config.set('augmentation.train.horizontal_flip', 0.7)
            config.set('augmentation.train.rotation', 10)
            config.set('augmentation.train.color_jitter.brightness', 0.2)
            config.set('augmentation.train.color_jitter.contrast', 0.2)
        logger.info(f"Augmentation strength set to: {args.augmentation_strength}")
    
    # Hardware configuration
    if args.device != 'auto':
        config.set('hardware.device', args.device)
    if args.mixed_precision:
        config.set('hardware.mixed_precision', True)
    if args.no_mixed_precision:
        config.set('hardware.mixed_precision', False)
    
    # Output configuration
    if args.output_dir:
        config.set('output.models_dir', os.path.join(args.output_dir, 'models'))
        config.set('output.logs_dir', os.path.join(args.output_dir, 'logs'))
    
    if args.experiment_name:
        # Add experiment name to output directories
        models_dir = config.get('output.models_dir')
        logs_dir = config.get('output.logs_dir')
        config.set('output.models_dir', os.path.join(models_dir, args.experiment_name))
        config.set('output.logs_dir', os.path.join(logs_dir, args.experiment_name))
        config.set('experiment_name', args.experiment_name)
    
    if args.save_every:
        config.set('output.checkpoint_every', args.save_every)

    # Class weight balancing (power scaling)
    if args.class_weight_mode:
        config.set('dataset.class_weight_mode', args.class_weight_mode)
    if args.class_weight_power_alpha is not None:
        config.set('dataset.class_weight_power_alpha', args.class_weight_power_alpha)
    
    return config

def determine_models_to_train(args, config: Config) -> List[str]:
    """Determine which models to train based on CLI arguments"""
    
    if args.model:
        # Validate specified models
        available_models = ModelFactory.SUPPORTED_MODELS
        invalid_models = [m for m in args.model if m not in available_models]
        
        if invalid_models:
            logger.error(f"Invalid models specified: {invalid_models}")
            logger.error(f"Available models: {available_models}")
            sys.exit(1)
        
        return args.model
    else:
        # Use models from config
        return config.get('models.variants', ['regnetx_008'])

def print_training_summary(config: Config, models_to_train: List[str]):
    """Print training configuration summary"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Models to train: {models_to_train}")
    logger.info(f"Dataset path: {config.get('dataset.path')}")
    logger.info(f"Batch size: {config.get('dataset.batch_size')}")
    logger.info(f"Input size: {config.get('dataset.image_size')}")
    logger.info(f"Phase 1 epochs: {config.get('training.phase1.epochs')}")
    logger.info(f"Phase 2 epochs: {config.get('training.phase2.epochs')}")
    logger.info(f"Phase 1 LR: {config.get('training.phase1.learning_rate')}")
    logger.info(f"Phase 2 LR: {config.get('training.phase2.learning_rate')}")
    logger.info(f"Mixed precision: {config.get('hardware.mixed_precision')}")
    logger.info(f"Device: {config.get('hardware.device')}")
    
    # Augmentation status
    horizontal_flip = config.get('augmentation.train.horizontal_flip', 0)
    rotation = config.get('augmentation.train.rotation', 0)
    if horizontal_flip > 0 or rotation > 0:
        logger.info(f"Augmentation: Enabled (flip: {horizontal_flip}, rotation: {rotation}°)")
    else:
        logger.info("Augmentation: Disabled")
    
    logger.info(f"Output directory: {config.get('output.models_dir')}")
    logger.info("="*60)

def main():
    """Enhanced main training function with RegNetX CLI support"""
    args = parse_arguments()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # List models and exit if requested
    if args.list_models:
        logger.info("Available models:")
        ModelFactory.list_available_models()
        
        # Also show parameter counts
        logger.info("\nRecommended RegNetX models by size:")
        logger.info("  Small  (~6M params): regnetx_006")
        logger.info("  Medium (~8M params): regnetx_008")
        logger.info("  Large  (~16M params): regnetx_016")
        return
    
    # Create runtime configuration
    logger.info("Loading configuration...")
    config = create_runtime_config(args, args.config)
    
    # Determine models to train
    models_to_train = determine_models_to_train(args, config)
    
    # Print configuration summary
    print_training_summary(config, models_to_train)
    
    # Exit if dry run
    if args.dry_run:
        logger.info("Dry run completed. Exiting without training.")
        return
    
    # Create output directories
    config.create_directories()
    
    # Setup device
    device = torch.device(config.get('hardware.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Initialize data processor
    logger.info("Loading and processing dataset...")
    data_processor = DataProcessor(config)
    
    # Initialize results aggregator
    results_aggregator = ResultsAggregator(config)
    
    # Train models
    successful_models = 0
    total_training_time = 0
    
    for i, model_name in enumerate(models_to_train, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING MODEL {i}/{len(models_to_train)}: {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            from src.train import train_single_model
            result = train_single_model(
                model_name=model_name,
                config=config,
                data_processor=data_processor,
                results_aggregator=results_aggregator
            )
            
            if result['success']:
                successful_models += 1
                total_training_time += result['training_time']
                logger.info(f"✅ {model_name} completed successfully!")
            else:
                logger.error(f"❌ {model_name} failed: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
    
    # Save comparison results
    if successful_models > 0:
        results_aggregator.save_comparison_table()
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Models trained successfully: {successful_models}/{len(models_to_train)}")
    if total_training_time > 0:
        logger.info(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
    
    if successful_models == 0:
        logger.error("❌ No models were trained successfully!")
        sys.exit(1)
    else:
        logger.info("✅ Training pipeline completed!")
        
        # Show best model if multiple were trained
        if successful_models > 1 and results_aggregator.results:
            best_result = max(results_aggregator.results, 
                            key=lambda x: x['age_f1'] + x['gender_f1'])
            logger.info(f"🏆 Best model: {best_result['model_name']}")
            logger.info(f"   Combined F1: {best_result['age_f1'] + best_result['gender_f1']:.4f}")

if __name__ == '__main__':
    main()
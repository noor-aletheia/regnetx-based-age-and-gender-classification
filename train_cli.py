import os
import sys
import torch
import logging
import argparse
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import Config
from dataset import DataProcessor
from models import ModelFactory
from trainer import TwoPhaseTrainer
from logger import TrainingLogger, ModelSaver, ResultsAggregator
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='RegNetX Age & Gender Classification Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--pause-between-models', type=int, default=0,
                       help='Pause (in seconds) between training each model (default: 0, no pause)')
    parser.add_argument('--model', '--models', nargs='*', 
                       help='Model(s) to train. If not specified, trains all models from config.')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    parser.add_argument('--data-path', 
                       help='Path to dataset (overrides config)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (overrides config)')
    parser.add_argument('--input-size', type=int,
                       help='Input image size (overrides config)')
    parser.add_argument('--num-workers', type=int,
                       help='Number of data  workers (overrides config)')
    
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
    
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                       default='auto', help='Device to use for training')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    
    parser.add_argument('--output-dir', 
                       help='Output directory (overrides config)')
    parser.add_argument('--experiment-name',
                       help='Experiment name (added to output paths)')
    parser.add_argument('--save-every', type=int,
                       help='Save checkpoint every N epochs')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration and exit without training')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    parser.add_argument('--class_weight_mode', choices=['none', 'auto', 'power', 'manual'],
                        help='Class weight calculation mode: none, auto, power, or manual (overrides config)')
    parser.add_argument('--class_weight_power_alpha', type=float,
                        help='Alpha value for power scaling of class weights (overrides config)')
    
    parser.add_argument('--age-classes', choices=['8-class', '4-class'],
                        help='Age classification scheme: 8-class (original) or 4-class (0-9, 10-29, 30-49, 50-70+)')
    parser.add_argument('--pretrained-model', type=str,
                        help='Path to pretrained model weights (.pth file) for transfer learning')
     
    return parser.parse_args()

def create_runtime_config(args, base_config_path: str) -> Config:
    """Create runtime configuration by merging CLI args with base config"""
    
    config = Config(base_config_path)
    
    if args.data_path:
        config.set('dataset.path', args.data_path)
    if args.batch_size:
        config.set('dataset.batch_size', args.batch_size)
    if args.input_size:
        config.set('dataset.image_size', args.input_size)
    if args.num_workers:
        config.set('dataset.num_workers', args.num_workers)
    
    if args.epochs:
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
    
    if args.no_augmentation:
        config.set('augmentation.train.motion_blur.prob', 0.0)
        config.set('augmentation.train.random_affine.degrees', 0)
        config.set('augmentation.train.color_jitter.brightness', 0.0)
        config.set('augmentation.train.color_jitter.contrast', 0.0)
        config.set('augmentation.train.color_jitter.saturation', 0.0)
        config.set('augmentation.train.color_jitter.hue', 0.0)
        config.set('augmentation.train.horizontal_flip', 0.0)
        config.set('augmentation.train.sharpening.prob', 0.0)
        config.set('augmentation.train.clahe.prob', 0.0)
        config.set('no_augmentation', True)
        logger.info("Data augmentation disabled")
    
    if args.device != 'auto':
        config.set('hardware.device', args.device)
    if args.mixed_precision:
        config.set('hardware.mixed_precision', True)
    if args.no_mixed_precision:
        config.set('hardware.mixed_precision', False)
    
    if args.output_dir:
        config.set('output.models_dir', os.path.join(args.output_dir, 'models'))
        config.set('output.logs_dir', os.path.join(args.output_dir, 'logs'))
    
    if args.experiment_name:
        models_dir = config.get('output.models_dir')
        logs_dir = config.get('output.logs_dir')
        config.set('output.models_dir', os.path.join(models_dir, args.experiment_name))
        config.set('output.logs_dir', os.path.join(logs_dir, args.experiment_name))
        config.set('experiment_name', args.experiment_name)
    
    if args.save_every:
        config.set('output.checkpoint_every', args.save_every)

    if args.class_weight_mode:
        config.set('dataset.class_weight_mode', args.class_weight_mode)
    if args.class_weight_power_alpha is not None:
        config.set('dataset.class_weight_power_alpha', args.class_weight_power_alpha)
    
    config.set('dataset.age_class_scheme', args.age_classes)
    
    if args.pretrained_model:
        config.set('models.pretrained_model_path', args.pretrained_model)
    
    return config

def determine_models_to_train(args, config: Config) -> List[str]:
    """Determine which models to train based on CLI arguments"""
    
    if args.model:
        available_models = ModelFactory.SUPPORTED_MODELS
        invalid_models = [m for m in args.model if m not in available_models]
        
        if invalid_models:
            logger.error(f"Invalid models specified: {invalid_models}")
            logger.error(f"Available models: {available_models}")
            sys.exit(1)
        
        return args.model
    else:
        return config.get('models.variants', [
            'regnet_200m', 'regnet_400m', 'regnet_600m', 'regnet_800m',
            'regnet_1600m', 'regnet_3200m', 'regnet_6400m'])

def print_training_summary(config: Config, models_to_train: List[str]):
    """Print training configuration summary"""
    logger.info("\n" + "="*60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Models to train: {models_to_train}")
    logger.info(f"Dataset path: {config.get('dataset.path')}")
    logger.info(f"Batch size: {config.get('dataset.batch_size')}")
    logger.info(f"Input size: {config.get('dataset.image_size')}")
    logger.info(f"Age class scheme: {config.get('dataset.age_class_scheme')}")
    logger.info(f"Phase 1 epochs: {config.get('training.phase1.epochs')}")
    logger.info(f"Phase 2 epochs: {config.get('training.phase2.epochs')}")
    logger.info(f"Phase 1 LR: {config.get('training.phase1.learning_rate')}")
    logger.info(f"Phase 2 LR: {config.get('training.phase2.learning_rate')}")
    logger.info(f"Mixed precision: {config.get('hardware.mixed_precision')}")
    logger.info(f"Device: {config.get('hardware.device')}")
    
    # Show pretrained model path if specified
    pretrained_model_path = config.get('models.pretrained_model_path')
    if pretrained_model_path:
        logger.info(f"Pretrained model: {pretrained_model_path}")
    else:
        logger.info("Pretrained model: Using ImageNet weights")
    
    # horizontal_flip = config.get('augmentation.train.horizontal_flip', 0)
    # if horizontal_flip > 0:
    #     logger.info(f"Augmentation: Enabled (flip: {horizontal_flip})")
    # else:
    #     logger.info("Augmentation: Disabled")
    
    logger.info(f"Output directory: {config.get('output.models_dir')}")
    logger.info("="*60)

def main():
    """Enhanced main training function with RegNetX CLI support"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list_models:
        logger.info("Available models:")
        logger.info("  regnet_200m, regnet_400m, regnet_600m, regnet_800m, regnet_1600m, regnet_3200m, regnet_6400m")
        logger.info("\nRecommended RegNetX models by size:")
        logger.info("  Small  (~0.2-0.8M params): regnet_200m, regnet_400m, regnet_600m, regnet_800m")
        logger.info("  Medium (~1.6-3.2M params): regnet_1600m, regnet_3200m")
        logger.info("  Large  (~4.0-6.4M params): regnet_6400m")
        return
    
    logger.info("Loading configuration...")
    config = create_runtime_config(args, args.config)
    
    models_to_train = determine_models_to_train(args, config)
    
    print_training_summary(config, models_to_train)
    
    if args.dry_run:
        logger.info("Dry run completed. Exiting without training.")
        return
    
    config.create_directories()
    
    device = torch.device(config.get('hardware.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    logger.info("Loading and processing dataset...")
    data_processor = DataProcessor(config)
    
    results_aggregator = ResultsAggregator(config)
    
    successful_models = 0
    total_training_time = 0
    
    import time
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

        # Pause between models if requested
        if args.pause_between_models and i < len(models_to_train):
            logger.info(f"Pausing for {args.pause_between_models} seconds before next model...")
            time.sleep(args.pause_between_models)
    
    if successful_models > 0:
        results_aggregator.save_comparison_table()
    
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
        
        if successful_models > 1 and results_aggregator.results:
            best_result = max(results_aggregator.results, 
                            key=lambda x: x['age_f1'] + x['gender_f1'])
            logger.info(f"🏆 Best model: {best_result['model_name']}")
            logger.info(f"   Combined F1: {best_result['age_f1'] + best_result['gender_f1']:.4f}")

if __name__ == '__main__':
    main()
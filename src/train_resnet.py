

import torch
from models_resnet import ResNetModelFactory
from src.models import create_loss_functions, create_optimizer, create_scheduler
from trainer import TwoPhaseTrainer
import time
import os


def train_single_model_resnet(model_name, config, data_processor, results_aggregator):
    import traceback
    result = {'success': False, 'model_name': model_name, 'training_time': 0}
    try:
        device = torch.device(config.get('hardware.device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        model = ResNetModelFactory.create_model(config, model_name)
        model = model.to(device)
        ResNetModelFactory.print_model_summary(model)

        train_loader, val_loader, test_loader = data_processor.get_dataloaders()
        age_class_names, gender_class_names = data_processor.get_class_names()
        age_class_weights, gender_class_weights = None, None
        if hasattr(data_processor, 'get_class_weights'):
            age_class_weights, gender_class_weights = data_processor.get_class_weights()

        trainer = TwoPhaseTrainer(
            model=model,
            config=config,
            age_class_weights=age_class_weights,
            gender_class_weights=gender_class_weights
        )
        trainer.initialize_metrics_calculator(age_class_names, gender_class_names)

        experiment_name = config.get('experiment_name', None)
        from logger import TrainingLogger, ModelSaver, ResultsAggregator
        training_logger = TrainingLogger(config, model_name, experiment_name=experiment_name)
        model_saver = ModelSaver(config)

        start_time = time.time()
        try:
            history = trainer.train(train_loader, val_loader)
        except Exception as e:
            import traceback
            print(f"Error during training: {str(e)}\n{traceback.format_exc()}")
            raise

        # Log each epoch's metrics
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

        training_time = time.time() - start_time

        # Evaluate on test set
        test_results = trainer.evaluate(test_loader, save_confusion_matrix=True)
        test_fps = test_results.get('performance', {}).get('overall_fps', 0.0)
        training_logger.log_test_results(test_results)

        # Save best model checkpoint
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

        # Compute GFLOPs if ptflops is available
        try:
            from ptflops import get_model_complexity_info
            dummy_input = torch.randn(1, 3, 224, 224)
            gflops, _ = get_model_complexity_info(model, (3, 224, 224), print_per_layer_stat=False, verbose=False)
            gflops = float(gflops.split(' ')[0]) if isinstance(gflops, str) else gflops
        except Exception:
            gflops = 0.0

        # Model info for aggregator
        model_info = ResNetModelFactory.get_model_info(model)
        training_config = {
            'phase1_epochs': config.get('training.phase1.epochs', 0),
            'phase2_epochs': config.get('training.phase2.epochs', 0),
            'augmentation_mode': config.get('augmentation.mode', {}),
            'phase1_lr': config.get('training.phase1.learning_rate', 0),
            'phase2_lr': config.get('training.phase2.learning_rate', 0),
            'phase1_wd': config.get('training.phase1.weight_decay', 0),
            'phase2_wd': config.get('training.phase2.weight_decay', 0),
            'scheduler': config.get('training.scheduler', 'unknown'),
            'class_weight_alpha': config.get('training.class_weight_power_alpha', 1.0),
            'loss_type': config.get('training.loss_type', 'unknown'),
            'batch_size': config.get('dataset.batch_size', 0),
            'age_class_scheme': config.get('dataset.age_class_scheme')
        }
        dataset_sizes = {
            'train': len(train_loader.dataset),
            'val': len(val_loader.dataset),
            'test': len(test_loader.dataset)
        }

        results_aggregator.add_model_results(
            model_name=model_name,
            test_results=test_results,
            training_time=training_time,
            model_info=model_info,
            training_config=training_config,
            dataset_sizes=dataset_sizes,
            gflops=gflops,
            fps=test_fps,
        )

        # Save training curves and history
        training_logger.close()

        result['success'] = True
        result['training_time'] = training_time
        result['age_f1'] = best_metrics['test_age_f1']
        result['gender_f1'] = best_metrics['test_gender_f1']
        result['model_name'] = model_name
        result['model_path'] = model_path
        result['test_results'] = test_results
    except Exception as e:
        result['error'] = str(e)
        print(f"Error in train_single_model_resnet: {str(e)}\n{traceback.format_exc()}")
    return result

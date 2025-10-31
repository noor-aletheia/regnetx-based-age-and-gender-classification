# MobileNet Age & Gender Classification

A comprehensive PyTorch pipeline for training MobileNet variants on age and gender classification from face images, with Docker support and advanced CLI features.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Python 3.9+ (for local development)

### Dataset Structure
Your dataset should follow this structure:

```
data/
├── train/
│   ├── groundtruth.csv    # Required: image,age,gender
│   ├── image1.jpg
│   └── image2.jpg
├── val/
│   ├── groundtruth.csv
│   ├── image1.jpg
│   └── image2.jpg
└── test/ (optional)
    ├── groundtruth.csv
    ├── image1.jpg
    └── image2.jpg
```

See `data/README.md` for detailed dataset format requirements.

### Basic Training

**Option 1: CLI Training (Recommended)**
```bash
# Validate dataset first
python3 validate_dataset.py --data-path ./data

# Train specific model
./train_cli.sh --model mobilenetv4_conv_small

# Train multiple models
./train_cli.sh --model mobilenetv3_small_075 mobilenetv4_conv_medium

# List available models
./train_cli.sh --list-models
```

**Option 2: Docker Training**
```bash
# Build and run with Docker Compose
docker-compose up

# With TensorBoard monitoring
docker-compose --profile tensorboard up
```

## 📋 Available Models

| Model | Size | Parameters | Description |
|-------|------|------------|-------------|
| `mobilenetv3_small_075` | Small | ~3.0M | MobileNetV3-Small with 0.75x width |
| `mobilenet_v3_small` | Small | ~2.9M | Efficient V3 small |
| `mobilenet_v3_large` | Medium | ~5.4M | Standard V3 large |
| `mobilenetv4_conv_small` | Small | ~6M | Latest V4 ConvNet small |
| `mobilenetv4_conv_medium` | Medium | ~11M | V4 ConvNet medium |
| `mobilenetv4_hybrid_medium` | Medium | ~11M | V4 Hybrid with attention |
| `mobilenetv4_conv_large` | Large | ~33M | V4 ConvNet large |

## 🛠️ CLI Usage

### Basic Commands
```bash
# Train all models (default)
./train_cli.sh

# Train specific model
./train_cli.sh --model mobilenetv4_conv_small

# Custom parameters without Docker rebuild
./train_cli.sh --model mobilenetv4_conv_small --epochs 25 --batch-size 32 --no-augmentation
```

### Advanced Usage
```bash
# Experiment with different augmentation
./train_cli.sh --model mobilenetv4_conv_small --augmentation-strength light

# Memory-efficient training
./train_cli.sh --model mobilenet_v3_small --batch-size 16 --cpu

# Custom experiment
./train_cli.sh --model mobilenetv4_conv_medium \
               --epochs 30 \
               --lr 0.001 \
               --experiment-name "optimized_training"
```

## 🏗️ Key Features

### Automatic Dataset Adaptation
- **Dynamic image size detection** - Automatically detects optimal input size
- **Flexible CSV parsing** - Intelligently finds age/gender/image columns
- **Any label scheme** - Works with your existing age/gender labels
- **Class imbalance handling** - Automatic class weight calculation

### Advanced Training
- **Two-phase training** - Frozen backbone → Full fine-tuning
- **Automatic phase transition** - Based on validation performance
- **Mixed precision training** - Faster training with memory savings
- **Comprehensive metrics** - Accuracy, precision, recall, F1, confusion matrices

### Production Ready
- **Docker containerization** - One-command deployment
- **CLI flexibility** - Change parameters without rebuilding
- **Model comparison** - Automatic benchmarking across variants
- **Export formats** - PyTorch (.pth) and ONNX models

## 📊 Output & Monitoring

### Generated Outputs
```
outputs/
├── models/              # Trained models (.pth, .onnx)
├── logs/               # Training logs and metrics
│   ├── confusion_matrix_age.png
│   ├── confusion_matrix_gender.png
│   └── training.log
├── tensorboard/        # TensorBoard logs
└── results/           # Model comparison tables
```

### TensorBoard Monitoring
```bash
# Start TensorBoard
tensorboard --logdir outputs/tensorboard
# Visit http://localhost:6006

# Or with Docker
docker-compose --profile tensorboard up
```

## ⚙️ Configuration

The system automatically detects:
- Image dimensions from your dataset
- Age/gender classes from CSV files  
- Optimal input size for training
- Column names in CSV files

Manual overrides available via CLI:
```bash
./train_cli.sh --input-size 448 --batch-size 32 --lr 0.002
```

## 🔧 Troubleshooting

### Common Issues
```bash
# CUDA out of memory
./train_cli.sh --batch-size 8 --no-mixed-precision

# Dataset validation errors
python3 validate_dataset.py --data-path ./data

# Slow training
./train_cli.sh --mixed-precision --batch-size 32
```

### Performance Expectations
- **Age Classification**: 75-85% accuracy across age classes
- **Gender Classification**: 90-95% accuracy  
- **Training Time**: 30-60 minutes per model (RTX 3080)

## � Project Structure

```
mobilenet/
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── dataset.py         # Dataset processing
│   ├── models.py          # Model architectures
│   ├── trainer.py         # Training logic
│   ├── logger.py          # Logging and metrics
│   └── train.py           # Main training script
├── data/                   # Dataset directory
├── outputs/               # Training outputs
├── config.yaml           # Training configuration
├── train_cli.py          # CLI training script
├── train_cli.sh          # CLI wrapper script
├── validate_dataset.py   # Dataset validation
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
└── requirements.txt      # Python dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 📚 References

- [MobileNets Paper](https://arxiv.org/abs/1704.04861)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)  
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [MobileNetV4 Paper](https://arxiv.org/abs/2404.10518)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Timm Library](https://github.com/huggingface/pytorch-image-models)
│   └── logger.py         # Logging and metrics
├── inference.py         # Inference script for deployment
├── evaluate.py          # Model evaluation and comparison
├── validate_data.py     # Dataset validation for 112x112 face crops
├── README.md            # This documentation
├── data/                # Dataset (user-provided)
│   ├── sorted_frames/   # Your 112x112 face crop dataset
│   └── groundtruth.csv  # Ground truth labels (optional)
└── outputs/              # Training outputs
    ├── models/           # Saved models (.pth, .onnx)
    ├── logs/             # Training logs and metrics
    └── tensorboard/      # TensorBoard logs
```

## 📊 Dataset Setup

### Face Crop Specifications
Your dataset should contain preprocessed face crops with these specifications:
- **Dimensions**: 112 × 112 pixels (square format)
- **Format**: JPEG (.jpg files)
- **Color**: RGB (3 channels)
- **Preprocessing**: Face-aligned crops with 1.25× padding factor
- **Processing Pipeline**:
  1. Face detection using Haar cascade
  2. Optional eye alignment using dlib landmarks
  3. 25% padding around detected face
  4. Boundary clipping to stay within image bounds
  5. Resize to 112×112 using OpenCV interpolation
  6. Save as JPEG format

### Dataset Structure
Organize your 112×112 face crops as follows:
```
data/sorted_frames/
├── Male/
│   ├── age 3-9/      # 112x112 male face crops aged 3-9
│   ├── age 10-19/    # 112x112 male face crops aged 10-19
│   ├── age 20-29/    # 112x112 male face crops aged 20-29
│   ├── age 30-39/    # 112x112 male face crops aged 30-39
│   ├── age 40-49/    # 112x112 male face crops aged 40-49
│   ├── age 50-59/    # 112x112 male face crops aged 50-59
│   ├── age 60-69/    # 112x112 male face crops aged 60-69
│   └── age 70+/      # 112x112 male face crops aged 70+
└── Female/
    ├── age 3-9/      # 112x112 female face crops aged 3-9
    ├── age 10-19/    # 112x112 female face crops aged 10-19
    ├── age 20-29/    # 112x112 female face crops aged 20-29
    ├── age 30-39/    # 112x112 female face crops aged 30-39
    ├── age 40-49/    # 112x112 female face crops aged 40-49
    ├── age 50-59/    # 112x112 female face crops aged 50-59
    ├── age 60-69/    # 112x112 female face crops aged 60-69
    └── age 70+/      # 112x112 female face crops aged 70+
```

### Data Validation
Before training, validate your dataset:

```bash
# Check dataset structure and image specifications
python validate_data.py

# Fix common issues (dry run)
python validate_data.py --dry-run

# Apply fixes to dataset
python validate_data.py --fix
```

The validation script will:
- Verify all images are 112×112 pixels
- Check file formats (JPEG preferred)
- Validate RGB color mode
- Report class distribution and balance
- Identify and optionally fix common issues

### CSV Ground Truth (Alternative)
If you have a CSV file, place it at `data/groundtruth.csv` with these columns:
```csv
image,folder,face_id,bbox,age_group,age_confidence,gender,gender_confidence
image1.jpg,Male,1,"[x,y,w,h]",age 20-29,0.95,Male,0.98
image2.jpg,Female,2,"[x,y,w,h]",age 30-39,0.87,Female,0.99
```

## ⚙️ Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Dataset Configuration
dataset:
  path: "/data/sorted_frames"
  csv_file: "/data/groundtruth.csv"
  batch_size: 64
  image_size: 224

# Model Configuration
models:
  variants: ["mobilenetv3_small_075", "mobilenet_v3_small", "mobilenet_v3_large"]
  num_age_classes: 8
  num_gender_classes: 2
  pretrained: true

# Training Configuration
training:
  phase1:  # Frozen backbone training
    epochs: 10
    learning_rate: 1e-3
    patience: 3
  phase2:  # Full fine-tuning
    epochs: 15
    learning_rate: 1e-4
    patience: 5

# Hardware Configuration
hardware:
  device: "cuda"
  mixed_precision: true
```

## 🏃‍♂️ Training Process

### Two-Phase Training Approach

**Phase 1: Frozen Backbone (Transfer Learning)**
- Freeze all MobileNet feature extractor layers
- Train only the classification heads
- Higher learning rate (1e-3)
- Early stopping when validation loss plateaus

**Phase 2: Full Fine-tuning**
- Unfreeze the entire model
- Lower learning rate (1e-4)
- Fine-tune all parameters
- Automatic transition from Phase 1

### Training Flow
1. **Model Creation**: Load pretrained MobileNet with dual classification heads
2. **Phase 1**: Train classification heads with frozen backbone
3. **Automatic Transition**: Switch to Phase 2 when validation loss plateaus
4. **Phase 2**: Fine-tune entire model end-to-end
5. **Evaluation**: Test on held-out test set
6. **Export**: Save best model and export to ONNX format

## 📊 Metrics and Evaluation

The pipeline tracks comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average precision
- **Recall**: Per-class and weighted average recall
- **F1-Score**: Per-class and weighted average F1-score
- **Confusion Matrix**: Visual representation of classification performance

### Output Files

After training, you'll find:

```
outputs/
├── models/
│   ├── mobilenetv3_small_075_best.pth  # Best PyTorch model
│   ├── mobilenetv3_small_075.onnx      # ONNX export
│   ├── mobilenet_v3_small_best.pth  # MobileNetV3-Small
│   ├── mobilenet_v3_small.onnx
│   ├── mobilenet_v3_large_best.pth  # MobileNetV3-Large
│   └── mobilenet_v3_large.onnx
├── logs/
│   ├── model_comparison.csv          # Comparison of all models
│   ├── training_summary.txt          # Human-readable summary
│   ├── mobilenetv3_small_075_training_history.csv
│   ├── mobilenetv3_small_075_training_curves.png
│   └── training.log                  # Full training log
└── tensorboard/
    ├── mobilenetv3_small_075/       # TensorBoard logs
    ├── mobilenet_v3_small/
    └── mobilenet_v3_large/
```

## 🐳 Docker Commands

### Basic Training
```bash
# Train all models
docker compose up --build

# Train with specific GPU
CUDA_VISIBLE_DEVICES=1 docker compose up --build

# Validate dataset before training
python validate_data.py
```

### TensorBoard Monitoring
```bash
# Start TensorBoard
docker compose --profile tensorboard up tensorboard

# Access at http://localhost:6006
```

### Development Mode
```bash
# Interactive development container
docker compose run --rm train bash

# Run specific script
docker compose run --rm train python src/train.py

# Validate data inside container
docker compose run --rm train python validate_data.py
```

### Data Management
```bash
# Check dataset statistics
python validate_data.py

# Fix common dataset issues (preview)
python validate_data.py --dry-run

# Apply fixes to dataset
python validate_data.py --fix

# Clean outputs directory
rm -rf outputs/models/* outputs/logs/* outputs/tensorboard/*
```

## 🔧 Customization

### Face-Specific Optimizations
The pipeline is optimized for your 112×112 face crops:

- **Automatic Resizing**: Images are resized from 112×112 to 224×224 for MobileNet compatibility
- **Face-Optimized Augmentation**:
  - Reduced rotation (5° max) since faces are pre-aligned
  - Reduced color jitter to preserve facial features
  - Horizontal flipping for data diversity
- **Input Validation**: Automatic verification of 112×112 dimensions

### Training Parameters for Face Data
```yaml
# Optimized augmentation for aligned face crops
augmentation:
  train:
    horizontal_flip: 0.5
    rotation: 5          # Reduced for aligned faces
    color_jitter:
      brightness: 0.1    # Reduced for faces
      contrast: 0.1      # Reduced for faces
      saturation: 0.1    # Reduced for faces
      hue: 0.05         # Reduced for faces
```

### Adding New Models
To add new model variants, modify `src/models.py`:

```python
def _create_backbone(self, model_name: str, pretrained: bool):
    if model_name == 'your_model':
        model = your_model_loader(pretrained=pretrained)
        backbone = model.features
    # ... rest of the implementation
```

### Custom Loss Functions
Modify `src/models.py` to add custom loss functions:

```python
def create_loss_functions(config, device, age_weights=None, gender_weights=None):
    # Add your custom loss function here
    custom_loss = YourCustomLoss()
    return custom_loss, gender_loss_fn
```

### Data Augmentation
Edit `config.yaml` to customize augmentation:

```yaml
augmentation:
  train:
    horizontal_flip: 0.5
    rotation: 15
    color_jitter:
      brightness: 0.3
      contrast: 0.3
```

## 🚀 Performance Tips

### GPU Optimization
- Use mixed precision training (enabled by default)
- Increase batch size if you have more GPU memory
- Use multiple GPUs with DataParallel (modify `src/trainer.py`)

### Speed Improvements
```yaml
# In config.yaml
dataset:
  num_workers: 8  # Increase for faster data loading
  batch_size: 128  # Increase if you have enough memory

hardware:
  mixed_precision: true
  compile_model: true  # Use torch.compile (PyTorch 2.0+)
```

### Memory Optimization
```yaml
dataset:
  batch_size: 32  # Reduce if running out of memory

training:
  phase1:
    epochs: 5  # Reduce epochs for faster training
  phase2:
    epochs: 10
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size in config.yaml
batch_size: 16

# Or use gradient accumulation
# (implement in src/trainer.py)
```

**CUDA Not Available**
```bash
# Check NVIDIA Docker runtime
docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi

# Ensure GPU support in docker-compose.yml
```

**Dataset Not Found**
```bash
# Check volume mounts
ls -la data/sorted_frames/

# Verify paths in config.yaml
dataset:
  path: "/data/sorted_frames"
```

**Permission Issues**
```bash
# Fix permissions
sudo chown -R $USER:$USER outputs/
chmod -R 755 outputs/
```

### Debugging Mode

Enable debug logging:
```yaml
# In config.yaml
logging:
  level: "DEBUG"
```

## 📈 Results Interpretation

### Model Comparison
The pipeline generates a comparison table ranking models by combined F1 score:

```
Model: mobilenet_v3_large
  Age - Acc: 0.8542, F1: 0.8234
  Gender - Acc: 0.9387, F1: 0.9345
  Combined F1: 1.7579

Model: mobilenet_v3_small
  Age - Acc: 0.8234, F1: 0.7892
  Gender - Acc: 0.9234, F1: 0.9198
  Combined F1: 1.7090
```

### Training Curves
Monitor training progress with automatically generated plots:
- Loss curves (train/validation)
- Accuracy curves (age/gender)
- F1 score progression
- Learning rate schedule
- Phase transitions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **torchvision** for pretrained MobileNet models
- **NVIDIA** for CUDA and Docker GPU support
- **TensorBoard** for training visualization

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section
2. Review logs in `outputs/logs/training.log`
3. Open an issue with:
   - Error message
   - Configuration file
   - System information
   - Dataset statistics

---

**Happy Training! 🎉**

Start your age and gender classification journey with:
```bash
docker compose up --build
```
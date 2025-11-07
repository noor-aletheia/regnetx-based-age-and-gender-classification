# RegNetX-Based Age and Gender Classification

A comprehensive PyTorch pipeline for age and gender classification from face images using RegNetX models. Features advanced data augmentation, two-phase training, Docker containerization, and robust ONNX export capabilities.

## 🎯 Key Features

- **Dual-Head RegNetX Architecture**: Simultaneous age and gender prediction
- **Advanced Data Pipeline**: Pre-augmentation + runtime augmentation hybrid approach  
- **Two-Phase Training**: Efficient frozen backbone → full fine-tuning methodology
- **Production-Ready**: Docker containerization with GPU acceleration
- **Robust Export**: Optimized ONNX conversion with similarity verification
- **Comprehensive Monitoring**: TensorBoard integration and detailed metrics

## 📊 Dataset Pipeline

### 1. Dataset Structure
```
expanded_merged_dataset/
├── train/           # ~97K images (pre-augmented + merged datasets)
├── val/             # ~12K images  
├── test/            # ~12K images
└── groundtruth.csv  # Unified labels: name,age,gender,path
```

### 2. Hybrid Augmentation Strategy
- **Pre-Augmentation**: Small custom datasets expanded 6x before merging
- **Runtime Augmentation**: 9 advanced techniques applied during training
  - Spatial: BBox crops, D4 transforms, grid distortion
  - Quality: Gaussian/motion blur, noise, downscaling  
  - Enhancement: CLAHE, grayscale conversion
  - Standard: Rotation, color jitter, flip, crop, erasing

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA GPU support
- ~8GB GPU memory for batch size 64
- CUDA-compatible GPU recommended

### Build Docker Image
```bash
docker build -t aletheiaai/noor:regnetx29-10-v2 .
```

### Training Command (200 Epochs with Advanced Augmentation)
```bash
docker run -d --gpus all \
  --name regnetx_training \
  --shm-size=8g \
  -v $(pwd)/expanded_merged_dataset:/app/expanded_merged_dataset \
  -v $(pwd)/outputs:/outputs \
  aletheiaai/noor:regnetx29-10-v2 \
  python train_cli.py \
  --data-path /app/expanded_merged_dataset \
  --model regnetx_016 \
  --age-classes 4-class \
  --phase1-epochs 10 \
  --phase2-epochs 190 \
  --lr 1e-3 \
  --lr2 1.5e-4 \
  --batch-size 64 \
  --weight-decay 1e-4 \
  --phase1-patience 6 \
  --phase2-patience 15 \
  --class_weight_mode power \
  --class_weight_power_alpha 2.0 \
  --augmentation-strength strong \
  --mixed-precision \
  --experiment-name regnetx_16_advanced_aug
```

## 📈 Training Workflow

### Phase 1: Frozen Backbone (10 epochs)
- **Learning Rate**: 1e-3 (fast head adaptation)
- **Trainable**: Classification heads only
- **Focus**: Quick convergence on new data distribution

### Phase 2: Full Fine-tuning (190 epochs) 
- **Learning Rate**: 1.5e-4 (stable optimization)
- **Trainable**: Complete network
- **Focus**: Deep feature adaptation with augmentation robustness

### Monitoring Training
```bash
# Check container status
docker ps

# View live training logs  
docker logs -f regnetx_training

# Monitor GPU usage
docker exec regnetx_training nvidia-smi

# Access container
docker exec -it regnetx_training bash
```

## 🏗️ Model Architecture

### Available Models
| Model | Parameters | GFLOPs | Description |
|-------|-----------|---------|-------------|
| `regnetx_006` | 6.5M | 0.6 | Lightweight, fast inference |
| `regnetx_008` | 7.5M | 0.8 | Balanced performance |
| `regnetx_016` | 9.5M | 1.6 | Best accuracy (recommended) |

### Age Classification Schemes
- **4-class**: 0-9, 10-29, 30-49, 50-70+ years (recommended)
- **8-class**: More granular age groups

## 📋 Output Files

```
outputs/
├── models/regnetx_16_advanced_aug/
│   ├── regnetx_016_best.pth          # Best PyTorch model
│   ├── regnetx_016_simplified.onnx   # FP32 ONNX
│   └── regnetx_016_fp16.onnx        # FP16 ONNX
├── logs/regnetx_16_advanced_aug/
│   ├── training_curves.png           # Loss/accuracy plots
│   ├── confusion_matrix_age.png      # Age classification matrix
│   ├── confusion_matrix_gender.png   # Gender classification matrix
│   ├── training_history.csv         # Epoch-by-epoch metrics
│   └── comprehensive_training_results.csv
└── tensorboard/regnetx_16_advanced_aug/  # TensorBoard logs
```
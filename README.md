# RegNetX Age & Gender Classification

A PyTorch pipeline for age and gender classification from face images using RegNetX models, with Docker and CLI support.

## Quick Start

### Prerequisites
- Docker (with NVIDIA GPU support recommended)
- Python 3.9+ (for local runs)

### Dataset Structure
```
data/
├── train/
│   ├── train_groundtruth.csv  # image,age,gender
│   ├── <images>.jpg
├── val/
│   ├── val_groundtruth.csv
│   ├── <images>.jpg
└── test/
    ├── test_groundtruth.csv
    ├── <images>.jpg
```

## Training

### CLI Example
```bash
python3 train_cli.py --model regnetx_006 --phase1-epochs 10 --phase2-epochs 20 --experiment-name my_exp --data-path ./data
```

### Docker Example
```bash
docker build -t regnetx-age-gender .
docker run --gpus all --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/outputs \
  regnetx-age-gender \
  python train_cli.py --model regnetx_006 --phase1-epochs 10 --phase2-epochs 20 --experiment-name my_exp --data-path /app/data
```

## Models
- `regnetx_006` (~6.5M params, ~0.6 GFLOPs)
- `regnetx_008` (~7.5M params, ~0.8 GFLOPs)
- `regnetx_016` (~9.5M params, ~1.6 GFLOPs)

## Outputs
- `outputs/models/` — Trained models (.pth, .onnx)
- `outputs/logs/` — Training logs, metrics, comparison tables
- `outputs/tensorboard/` — TensorBoard logs

## Monitoring
```bash
tensorboard --logdir outputs/tensorboard
```

## Features
- Two-phase training (frozen backbone, then fine-tuning)
- Mixed precision support
- Automatic class weight calculation
- TensorBoard and CSV logging
- Docker-ready for reproducible runs

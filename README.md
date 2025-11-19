# RegNetX-Based Age and Gender Classification

## Repository Structure

| File/Folder         | Description |
|---------------------|-------------|
| `src/`              | Source code for models, training, logger, and utilities |
| `train_cli.py`      | Main CLI script for training models |
| `train_cli.sh`      | Bash script for launching training with arguments |
| `inference.py`      | Script for running inference and evaluation |
| `export.py`         | Script for ONNX export and model conversion |
| `regnet.py`         | RegNetX backbone model definitions |
| `reglayers.py`      | Custom layers and wrappers for RegNetX |
| `regnet_weights/`   | Pretrained RegNet weights (downloaded from upstream) |
| `outputs/`          | Output directory for models, logs, and tensorboard |
| `config.yaml`       | Main configuration file for experiments |
| `Dockerfile`        | Docker build instructions for reproducible environment |
| `requirements.txt`  | Python dependencies for the project |
| `quantize/`         | Scripts and outputs for model quantization and deployment |


## Input Format

- **Image Input:** All images should be RGB, 224x224 pixels, normalized with mean = [0.498, 0.498, 0.498] and std = [0.498, 0.498, 0.498].
- **CSV Labels:** The main CSV file (e.g., `groundtruth.csv`) should have columns: `name`, `age`, `gender`, `path`.
  - `name`: Image filename
  - `age`: Integer age or age class/range
  - `gender`: 'male' or 'female'
  - `path`: Relative or absolute path to image (optional if images are in standard folders)


## Dataset Structure

```
expanded_merged_dataset/
├── train/           # Training images 
├── val/             # Validation images
├── test/            # Test images
└── groundtruth.csv  # Unified labels: name, age, gender, path
```

- `train/`, `val/`, `test/`: Folders containing images for each split.
- `groundtruth.csv`: CSV file with unified labels for all images.


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

## Training Workflow

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
# View live training logs  
docker logs -f regnetx_training

# Access container
docker exec -it regnetx_training bash
```

## RegNet Weights and Backbone source

The RegNet weights used in this project were obtained from:
https://github.com/yhhhli/RegNet-Pytorch/tree/master/ImageNet/models


## Model Architecture

- **Dual-Head Output:** The RegNetX backbone is extended with two classification heads:
  - **Age Head:** Outputs logits for age classes (4 or 8 classes, configurable)
  - **Gender Head:** Outputs logits for gender (2 classes: male, female)
  - Both heads are trained jointly for multi-task learning, improving overall performance.

### Available Models
| Model         | Parameters | Description |
|---------------|------------|-------------|
| `regnet_200m` | 2.7M       | Smallest, fastest |
| `regnet_400m` | 4.2M       | Small, fast |
| `regnet_600m` | 6.2M       | Small, balanced |
| `regnet_800m` | 7.3M       | Small, balanced |
| `regnet_1600m`| 9.2M       | Medium, best trade-off |
| `regnet_3200m`| 15.3M      | Medium-large |
| `regnet_4000m`| 20.6M      | Large |
| `regnet_6400m`| 25.5M      | Largest, highest accuracy |

### Age Classification Schemes
- **4-class**: 0-9, 10-29, 30-49, 50-70+ years (recommended)
- **8-class**: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

## Output Files

```
outputs/
├── models/<experiment_name>/
│   ├── <model_name>_best.pth          # Best PyTorch model
│   └── <model_name>.onnx   # Simplified ONNX with preprocessing
├── logs/<experiment_name>/
│   ├── training_curves.png           # Loss/accuracy plots
│   ├── confusion_matrix_age.png      # Age classification matrix
│   ├── confusion_matrix_gender.png   # Gender classification matrix
│   ├── training_history.csv         # Epoch-by-epoch metrics
│   └── comprehensive_training_results.csv
└── tensorboard/<experiment_name>/  # TensorBoard logs
```
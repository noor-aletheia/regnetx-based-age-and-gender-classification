# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt \
    && pip install tensorboard \
    && mkdir -p /outputs/models /outputs/logs /outputs/tensorboard


# Copy source code
COPY src/ ./src/
COPY config.yaml .
COPY train_cli.py .
COPY train_cli.sh .


# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "src/train.py"]
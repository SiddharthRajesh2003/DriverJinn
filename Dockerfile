# Base image with CUDA 11.8 and cuDNN
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.7.1 torchvision=0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install torch-geometric==2.6.1

# Install DGL with CUDA 11.8
RUN pip install dgl==2.1.0

# Copy requirements file
COPY requirements.txt .

# Install remaining python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .


# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "model/hyperparameter_search.py", "--help"]
# Dedicated Dockerfile for GraphSVX explanations
FROM ubuntu:22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev \
    build-essential git curl wget ca-certificates \
    libopenmpi-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set workdir
WORKDIR /graphsvx

# Clone GraphSVX repository
RUN git clone https://github.com/AlexDuvalinho/GraphSVX.git .

# Upgrade pip and install PyTorch and dependencies (CUDA 12.1 version for GPU support)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install torch-geometric and dependencies (CUDA 12.1 wheels)
RUN python3 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
    python3 -m pip install torch-geometric

# Install other Python dependencies for GraphSVX
RUN python3 -m pip install -r requirements.txt

# Ensure CUDA-enabled PyTorch remains installed (GraphSVX requirements may pin CPU wheels)
RUN python3 -m pip install --upgrade \
    torch==2.3.1+cu121 \
    torchvision==0.18.1+cu121 \
    torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --upgrade \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
    python3 -m pip install --upgrade torch-geometric

# Copy and install application requirements
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt

# Application requirements may drag in a newer torch, which breaks the PyG extension
# wheels built for 2.3.1+cu121. Re-pin the stack afterwards and verify it loads.
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir "numpy<2"
RUN python3 -c "import torch, torch_scatter, torch_sparse, torch_cluster, torch_geometric; print('torch', torch.__version__, 'pyg', torch_geometric.__version__)"

# Optional: install Jupyter for interactive use
RUN python3 -m pip install jupyter

# Entrypoint for running GraphSVX scripts
ENTRYPOINT ["/bin/bash"]

#!/bin/bash
#
# Sets up a robust and reproducible Python environment for the project.
# 1. Installs essential system packages including bc for shell calculations.
# 2. Cleans old, potentially incompatible data cache files.
# 3. Upgrades pip and installs specific, known-compatible Python libraries.

echo "--- Setting up environment ---"

# --- Step 1: Install system-level dependencies ---
echo "Installing build-essential, python3-dev, pybind11-dev, and bc..."
sudo apt-get update
sudo apt-get install -y build-essential python3-dev pybind11-dev bc

# --- Step 2: Clean old cache files ---
echo "Cleaning old *.pkl cache files from data directories..."
find . -type f -name "*.pkl" -delete

# --- Step 3: Upgrade pip and install core Python libraries ---
echo "Upgrading pip and installing compatible core libraries..."
pip install --upgrade pip
pip install networkx==3.2.1 numpy==1.26.4 scikit-learn==1.4.2 tqdm

# --- Step 4: Install a stable version of PyTorch (CPU version) ---
echo "Installing a stable version of PyTorch..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# --- Step 5: Install PyG libraries from the official wheel index ---
echo "Installing PyTorch Geometric libraries (torch-scatter, etc.)..."
pip install torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python3 -c 'import torch; print(torch.__version__.split("+")[0])').html

# --- Step 6: Install remaining project-specific packages ---
echo "Installing other required packages..."
pip install -q pipreqs
pipreqs . --force
pip install -r requirements.txt

echo "--- Environment setup complete. ---"
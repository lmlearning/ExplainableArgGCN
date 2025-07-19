#!/bin/bash
#
# Sets up the Python environment for the project.
# 1. Installs pipreqs.
# 2. Generates requirements.txt from .py files in the current directory.
# 3. Installs all required packages.

echo "--- Setting up Python environment ---"

# Install pipreqs if not already installed
pip install -q pipreqs

# Generate requirements.txt, overwriting any existing file
echo "Generating requirements.txt..."
pipreqs . --force

# Manually add packages that pipreqs might miss, like torch-scatter
echo "torch_scatter" >> requirements.txt

# Install packages from the generated requirements file
echo "Installing required packages..."
pip install -r requirements.txt

echo "--- Environment setup complete. ---"
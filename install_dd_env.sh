#!/bin/bash
set -euo pipefail

#========================================
# TEMPLATE FOR CONDA ENV INSTALL
# remove and edit file to represent specific project
#========================================

echo "====================================================="
echo "==> RUNNING CONDA ENVIRONMENT CREATION SCRIPT"

#============ Get Environment Name ==========
echo "==> Enter the environment name (default: default):"
read -rp ">>> " env_name
env_name=${env_name:-default}
echo "==> Using environment name: $env_name"


#============ Create the environment and install mamba =====

echo "==> Creating Conda environment"
echo
echo "==> Attempting installation with Mamba"
if ! command -v mamba &> /dev/null; then
    echo "[[Mamba is not installed]]"
    echo "==> Proceeding to installation with conda..."
    conda create --name $env_name -y || {
        echo "[[Failed to create the environment. Please check if conda is installed properly and added to PATH]]"
        exit 1
    }
else
    mamba env create --name $env_name -y || {
        echo "[[Failed to create the environment. Please check if conda is installed properly and added to PATH]]"
        exit 1
    }
fi

echo "==> Activating environment for installing libraries"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_name" || {
    echo "[[Failed to activate the environment. Please check if the environment was created successfully.]]"
    exit 1
}
echo "==> Installing mamba"
conda install mamba -y

# Installing Python and Pip
echo ">>> Installing Python 3.10 and Pip"
mamba install python=3.10 pip=24.0 -y

# Additional libraries

# Ninja
echo ">>> Installing Ninja for detectron"
mamba install ninja -y

#Installing other dependencies

echo ">>> Set 1 of dependencies"
mamba install -y numpy==1.26.4 pandas jupyterlab pillow tesserocr
echo ">>> Set 2 of dependencies"
mamba install -y matplotlib opencv transformers json-repair poppler
echo ">>> set 3 of dependencies"
mamba install -y pydantic pytesseract spacy protobuf sentencepiece

# Pytorch
echo ">>> Installing Torch and Torchvision with support for CUDA 12.9"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Detectron2
echo ">>> Installing detectron2"
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo ">>> Set 4 :- pip dependencies"
pip install pdf2image natsort fuzzywuzzy python-Levenshtein accelerate

echo ">>> Installing TaxoNerd"
pip install taxonerd

# DeepDoctection
pip install deepdoctection[pt]

echo "==> Environment '$env_name' created successfully."
echo "====================================================="

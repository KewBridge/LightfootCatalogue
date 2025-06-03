#!/bin/bash
set -euo pipefail


# Install the requirements from the requirements.yml file
echo "====================================================="
echo "==> Installing requirements from requirements.yml..."

# ============ Environment name ============

read -rp "==> Enter the environment name (default: lightfoot): " env_name
env_name=${env_name:-lightfoot}
echo "==> Using environment name: $env_name"

# ============ Install requirements ============

if ! command -v mamba &> /dev/null; then
    echo "[[Mamba is not installed.]]"
    echo "==> Attempting install with conda"
    conda env create -f requirements.yml -n "$env_name" -y || {
        echo "[[Failed to create the environment. Please check the requirements.yml file.]]"
        exit 1
    }
else
    echo "[[Mamba is installed.]]"
    echo "==> Installing with Mamba"
    mamba env create -f requirements.yml -n "$env_name" -y || {
        echo "[[Failed to create the environment. Please check the requirements.yml file.]]"
        exit 1
    }
fi

mamba env create -f requirements.yml -n "$env_name" -y || {
    echo "[[Failed to create the environment. Please check the requirements.yml file.]]"
    exit 1
}

echo "===> Requirements file installed successfully."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_name" || {
    echo "[[Failed to activate the environment. Please check if the environment was created successfully.]]"
    exit 1
}

# Installing pytorch with CUDA 12.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "==> Environment '$env_name' created successfully."


# ============ Additional installations ============
echo "====================================================="
echo "==> Installing additional packages..."
echo
echo "==> Installing taxonerd model: en_ner_eco_md"

pip install https://github.com/nleguillarme/taxonerd/releases/download/v1.5.4/en_ner_eco_md-1.1.0.tar.gz

echo "======================================================"
echo 
echo "     Set up complete. You can run the following:      "
echo "   Activate Environment : conda activate $env_name    "
echo " Deactivate Environment : conda deactivate            "
echo 
echo "======================================================"

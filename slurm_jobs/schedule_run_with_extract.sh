#!/bin/bash
#
#SBATCH --job-name="lightfootcat test job"
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --export=all

source activate lcat-step
echo ">>> Activated Environment <<<"
python run.py ./images/lightfootcat ./prompts/ost_prompt.yaml lightfootcat  --save-path ./outputs/lightfootcat_with_extract/

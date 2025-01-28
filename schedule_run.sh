#!/bin/bash
#
#SBATCH --job-name="lightfootcat test job"
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --export=all

source activate lcat-step
python run.py ~/scratch/private/LightfootCatalogue/images ~/scratch/private/LightfootCatalogue/prompts/ost_prompt.yaml lightfootcat_full --save-path ~/scratch/private/LightfootCatalogue/outputs/lightfootcat/

#!/bin/bash
#
#SBATCH --job-name="lightfootcat test job"
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --export=all

source activate dd_lcat
echo ">>> Activated Environment <<<"
python run.py ./images/lightfootcat ./prompts/ost_prompt.yaml lightfootcat_full --temp-text ./outputs/lightfootcat/temp.txt  --save-path ./outputs/lightfootcat/

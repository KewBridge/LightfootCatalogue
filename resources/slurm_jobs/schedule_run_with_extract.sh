#!/bin/bash
#
#SBATCH --job-name="lightfootcat test job"
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --export=all

CATALOGUE=${CATALOGUE_NAME:-lightfootcat}

source activate lcat-step
echo ">>> Activated Environment <<<"
echo "Running job for dataset: $CATALOGUE"

case $CATALOGUE in
    lightfootcat)
        python run.py ./resources/images/lightfootcat/images/ ./resources/prompts/lightfootcat_prompt.yaml lightfootcat  --save-path ./outputs/lightfootcat_with_extract/
        ;;
    hanbury)
        python run.py ./resources/images/hanbury ./resources/prompts/hanbury_prompt.yaml hanbury  --save-path ./outputs/hanbury_with_extract/
        ;;
    *)
        echo "CATALOGUE NOT RECOGNISED!"
        echo "ENDING RUN..."
        ;;
esac


echo "JOB COMPLETED"

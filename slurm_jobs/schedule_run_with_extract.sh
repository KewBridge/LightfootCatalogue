#!/bin/bash
#
#SBATCH --job-name="lightfootcat test job"
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --export=all

CATALOGUE=${CATALOGUE_NAME:-lightfootcat}

source activate lcat-step
echo ">>> Activated Environment <<<"
echo "Running job for dataset: $CATALOGUE"

case $CATALOGUE in
    lightfootcat)
        python run.py ./images/lightfootcat/images/ ./prompts/lightfootcat_prompt.yaml lightfootcat  --save-path ./outputs/lightfootcat_with_extract/
        ;;
    hanbury)
        python run.py ./images/hanbury ./prompts/hanbury_prompt.yaml hanbury  --save-path ./outputs/hanbury_with_extract/
        ;;
    *)
        echo "CATALOGUE NOT RECOGNISED!"
        echo "ENDING RUN..."
        ;;
esac


echo "JOB COMPLETED"

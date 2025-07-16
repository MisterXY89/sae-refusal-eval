#!/usr/bin/env bash
#SBATCH --job-name=eval-grid-SAE
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grid_eval_%A_%a.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END
#SBATCH --array=0-499%10

# NOTE: The range --array=0-399 runs 400 jobs for your configs.
# The '%10' limits the number of simultaneously running jobs to 10.

set -e

# --- Configuration ---
# Updated config file name
CONFIG_FILE="evaluation/feature_grid_configs_135M.json"
TARGET_MODEL="EleutherAI/sae-SmolLM2-135M-64x"

# --- Early-Exit Check ---
# Read the sparse_model for the current job's config. The -r flag gets the raw string without quotes.
SPARSE_MODEL=$(jq -r ".[${SLURM_ARRAY_TASK_ID}].sparse_model" $CONFIG_FILE)

# Check if the model does not match the target model.
if [[ "$SPARSE_MODEL" != "$TARGET_MODEL" ]]; then
    echo "Job ${SLURM_ARRAY_TASK_ID}: Skipping model '${SPARSE_MODEL}'."
    exit 0 # Exit successfully without doing any work.
fi

COEFFICIENT=$(jq ".[${SLURM_ARRAY_TASK_ID}].steering_coefficients[0]" $CONFIG_FILE)

# Check if the coefficient is not 1. Note the quotes for string comparison.
if [[ "$COEFFICIENT" == "1" ]]; then
   echo "Job ${SLURM_ARRAY_TASK_ID}: Skipping evaluation for C=${COEFFICIENT}."
    exit 0 # Exit successfully without doing any work
fi

# --- Environment Setup ---
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

# Navigate to the project's root directory
cd /home/tilman.kerl/mech-interp/src/

# --- Job Execution ---
echo "--- Starting evaluation for config index: ${SLURM_ARRAY_TASK_ID} ---"

# The Python script to execute
# It's good practice to use 'python -u' for unbuffered output in logs
python -u  -m evaluation.single_steer_feature_grid \
    --config_path "$CONFIG_FILE" \
    --config_index "${SLURM_ARRAY_TASK_ID}"

echo "--- Evaluation for config index ${SLURM_ARRAY_TASK_ID} finished. ---"
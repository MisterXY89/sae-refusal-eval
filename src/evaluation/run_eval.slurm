#!/usr/bin/env bash
#SBATCH --job-name=eval-smollm2-SAE
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_eval_%A_%a.out
#SBATCH --array=0-1
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END

set -e

# ALL SAEs:
# /home/tilman.kerl/mech-interp/src/train/MIX/checkpoints/smollm2-sparsify-EQ-50M-token-6_25-layers-32-expansion-64-k
# /home/tilman.kerl/mech-interp/src/train/MIX/checkpoints/smollm2-sparsify-INS-50M-token-6_25-layers-32-expansion-64-k
# /home/tilman.kerl/mech-interp/src/train/MIX/checkpoints/smollm2-sparsify-PRE-50M-token-6_25-layers-32-expansion-64-k
# /home/tilman.kerl/mech-interp/src/train/LMSYS/checkpoints/smollm2-sparsify-lmsys-50M-token-6_25-layers-32-expansion-64-k

# -------- checkpoint-specific parameters (index-aligned) ----------
SPARSE_MODELS=(
    sae-SmolLM2-135M-64x
)
PRE_TRAINED_MODEL="HuggingFaceTB/SmolLM2-135M"

HOOKPOINTS=(
    layers.6
    layers.24
)

# each element may be a space-separated list of indices
FEATURE_INDICES=(
    "928 730 606 376 125 703168"
    "421 "
)

# corresponding lists of coefficients
STEERING_COEFS=(
    "16.0 15.0 15.0 15.0 16.0"
    "22.0"
)

# load conda and activate environment
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

# select entry based on SLURM_ARRAY_TASK_ID
ID=${SLURM_ARRAY_TASK_ID}
SPARSE_MODEL=${SPARSE_MODELS[$ID]}
HOOKPOINT=${HOOKPOINTS[$ID]}
FEATURE_INDEX_LIST=${FEATURE_INDICES[$ID]}
STEERING_COEF_LIST=${STEERING_COEFS[$ID]}
RUN_TAG=$(basename "${SPARSE_MODEL%/}")

# prepare steer_path for this run
STEER_PATH="/home/tilman.kerl/mech-interp/src/steer_configs/${RUN_TAG}-${HOOKPOINT}.pt"
mkdir -p "$(dirname "$STEER_PATH")"

cd /home/tilman.kerl/mech-interp/src

# invoke evaluation with multiple feature indices and coefficients
python -m evaluation.run_eval \
  --model_type steered \
  --pretrained "$PRE_TRAINED_MODEL" \
  --steer_path "$STEER_PATH" \
  --loader sparsify \
  --action add \
  --sparse_model "$SPARSE_MODEL" \
  --hookpoint "$HOOKPOINT" \
  --feature_indices $FEATURE_INDEX_LIST \
  --steering_coefficients $STEERING_COEF_LIST \
  --tasks mmlu,hellaswag \
  --batch_size 64 \
  --device cuda:0

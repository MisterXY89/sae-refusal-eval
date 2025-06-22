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
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=ALL

# S B A T C H 2               # <-- 3 parallel tasks --array=0-2

set -e

# -------- checkpoint-specific parameters (index-aligned) ----------
SPARSE_MODELS=(
  EleutherAI/sae-SmolLM2-135M-64x
)

# HuggingFaceTB/SmolLM2-135M
PRE_TRAINED_MODEL="HuggingFaceTB/SmolLM2-135M"

HOOKPOINTS=(layers.24.mlp)
FEATURE_INDICES=(25)

# -------- payload -------------------------------------------------
ID=${SLURM_ARRAY_TASK_ID}
SPARSE_MODEL=${SPARSE_MODELS[$ID]}
HOOKPOINT=${HOOKPOINTS[$ID]}
FEATURE_INDEX=${FEATURE_INDICES[$ID]}
RUN_TAG=$(basename "${SPARSE_MODEL%/}")

cd /home/tilman.kerl/mech-interp/src
python -m evaluation.run_eval \
  --model_type steered \
  --pretrained "$PRE_TRAINED_MODEL" \
  --loader sparsify \
  --action add \
  --sparse_model "$SPARSE_MODEL" \
  --hookpoint "$HOOKPOINT" \
  --feature_index "$FEATURE_INDEX" \
  --steering_coefficient 13.0 \
  --tasks mmlu,hellaswag \
  --batch_size 64 \
  --device cuda:0 \
  --wandb_project MA-sae-eval

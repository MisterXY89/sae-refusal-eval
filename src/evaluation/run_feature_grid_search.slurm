#!/usr/bin/env bash
#SBATCH --job-name=SAE-feat-id
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/full_eval_%A_%a.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END

set -e

# -------- checkpoint-specific parameters (index-aligned) ----------
SPARSE_MODELS=(
  EleutherAI/sae-SmolLM2-135M-64x
)

# HuggingFaceTB/SmolLM2-135M
PRE_TRAINED_MODEL="HuggingFaceTB/SmolLM2-135M"


source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

# -------- payload -------------------------------------------------
ID=${SLURM_ARRAY_TASK_ID}
SPARSE_MODEL=${SPARSE_MODELS[$ID]}
HOOKPOINT=${HOOKPOINTS[$ID]}
FEATURE_INDEX=${FEATURE_INDICES[$ID]}
RUN_TAG=$(basename "${SPARSE_MODEL%/}")

cd /home/tilman.kerl/mech-interp/src
python -m evaluation.grid_feature_search \
  --sparse_model "$SPARSE_MODEL" \
  --pretrained "$PRE_TRAINED_MODEL" \
  --layer 24 \
  --exp_factor 64
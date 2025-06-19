#!/usr/bin/env bash
#SBATCH --job-name=eval-smollm2-SAE
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2               # <-- 3 parallel tasks
#SBATCH --output=logs/full_eval_%A_%a.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=ALL

set -e

# -------- checkpoint-specific parameters (index-aligned) ----------
SPARSE_MODELS=(
  /home/tilman.kerl/mech-interp/src/train/PRE/checkpoints/ckpt/
  /home/tilman.kerl/mech-interp/src/train/INS/checkpoints/ckpt/
  /home/tilman.kerl/mech-interp/src/train/EQ/checkpoints/ckpt/
)

HOOKPOINTS=(layers.18 layers.6 layers.25)        # example
FEATURE_INDICES=(17 42 5)                        # example

# -------- payload -------------------------------------------------
ID=${SLURM_ARRAY_TASK_ID}
SPARSE_MODEL=${SPARSE_MODELS[$ID]}
HOOKPOINT=${HOOKPOINTS[$ID]}
FEATURE_INDEX=${FEATURE_INDICES[$ID]}
RUN_TAG=$(basename "${SPARSE_MODEL%/}")

python run_eval.py \
  --model_type steered \
  --pretrained HuggingFaceTB/SmolLM2-135M \
  --loader sparsify \
  --action add \
  --sparse_model "$SPARSE_MODEL" \
  --hookpoint "$HOOKPOINT" \
  --feature_index "$FEATURE_INDEX" \
  --steering_coefficient 3.0 \
  --tasks mmlu,hellaswag \
  --batch_size 16 \
  --device cuda:0 \
  --wandb_project MA-sae-eval \
  --run_name "eval-${RUN_TAG}"

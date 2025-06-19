#!/usr/bin/env bash
#SBATCH --job-name=grid-smollm2-SAE
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/sparsify_smol2_grid_%j.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=ALL

set -e

# --- 1. parameter grids -------------------------------------------------------
TOTAL_MAX=6557946
# ~419 M … 50 M tokens
MAX_EXAMPLES_LIST=(6557940 3906200 1953100 781250)          
EXPANSION_FACTOR_LIST=(8 16 32)

MODEL="HuggingFaceTB/SmolLM2-135M"
DATASET_BASE="MisterXY89"
TRAIN_SPLITS=(pre_dominant_70_30 instruction_dominant_30_70 balanced_50_50)
# short tags for names
SPLIT_TAGS=(PRE INS EQ)

LAYERS=(6 25)
# helper string for the run-name
LAYERS_TAG="$(IFS=_; echo "${LAYERS[*]}")"   # -> "6_25"
"
NUM_PROC=4

K=64
BATCH_SIZE=48
GRAD_ACC_STEPS=4
CTX_LEN=64
WANDB_FREQ=100

# --- 2. environment -----------------------------------------------------------
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

# --- 3. main loop -------------------------------------------------------------
for s_idx in "${!TRAIN_SPLITS[@]}"; do
    SPLIT_NAME="${TRAIN_SPLITS[$s_idx]}"
    SPLIT_TAG="${SPLIT_TAGS[$s_idx]}"
    DATASET="${DATASET_BASE}/${SPLIT_NAME}"

    echo "################  DATASET = ${DATASET}  ################"

    for MAX_EX in "${MAX_EXAMPLES_LIST[@]}"; do
        MAX_EX_MB=$(((MAX_EX * 64) / 1000000))              # tokens → MB

        for EXPANSION_FACTOR in "${EXPANSION_FACTOR_LIST[@]}"; do
            RUN_NAME="smollm2-sparsify-${SPLIT_TAG}-${MAX_EX_MB}M-token-${LAYERS_TAG}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

            echo "==> Run ${RUN_NAME}"

            torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
              --split "train[:${TOTAL_MAX}]" \
              --max_examples ${MAX_EX} \
              --expansion_factor ${EXPANSION_FACTOR} \
              --k ${K} \
              --batch_size ${BATCH_SIZE} \
              --grad_acc_steps ${GRAD_ACC_STEPS} \
              --ctx_len ${CTX_LEN} \
              --layers "${LAYERS[@]}" \
              --data_preprocessing_num_proc ${NUM_PROC} \
              --log_to_wandb \
              --run_name ${RUN_NAME} \
              --wandb_log_frequency ${WANDB_FREQ} \
              --distribute_modules

            echo "<-- finished ${RUN_NAME}"
        done
    done
done

echo "--- All runs completed. Job finished. ---"

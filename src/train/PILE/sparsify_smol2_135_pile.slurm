#!/usr/bin/env bash

#SBATCH --job-name=grid-smollm2-SAE
# alt: a100, a100s
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G  #32/48 for a40
# IMPORTANT: Time is for ALL runs combined. 2 MAX_EX * 3 factors = 6 runs.
#SBATCH --time=10:00:00
#SBATCH --output=logs/sparsify_smol2_grid_%j.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=ALL

# Exit immediately if a command exits with a non-zero status.
set -e

# === Parameter Lists for Grid Search ===
# Add any number of example counts you want to test here
TOTAL_MAX=6557946
MAX_EXAMPLES_LIST=(6557940 3906200 1953100 781250) # Corresponds to ~419M ~250M and ~125M and 50M tokens
EXPANSION_FACTOR_LIST=(8 16 32)

# === Constant Parameter Definitions ===
MODEL="HuggingFaceTB/SmolLM2-135M"
DATASET="monology/pile-uncopyrighted"
LAYERS="6 25"
NUM_PROC=4

# === Constant Hyperparameters ===
K=64
BATCH_SIZE=48   # 32 for a40, 48 for a100
GRAD_ACC_STEPS=4
CTX_LEN=64

# === Constant Logging ===
WANDB_FREQ=100

# === Environment setup ===
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=hf_XXXXXXX
export HF_HOME=/share/tilman.kerl/huggingface


# === Main Execution Loop ===
# Outer loop for MAX_EX
for max_ex_val in "${MAX_EXAMPLES_LIST[@]}"; do
    # Set MAX_EX for this batch of runs
    MAX_EX=${max_ex_val}
    # IMPORTANT: Recalculate MAX_EX_MB inside the loop
    MAX_EX_MB=$(( (MAX_EX * 64) / 1000000 ))

    echo "#####################################################"
    echo "### Starting Set of Runs for MAX_EX = ${MAX_EX} (${MAX_EX_MB}M tokens) ###"
    echo "#####################################################"

    # Inner loop for EXPANSION_FACTOR
    for factor in "${EXPANSION_FACTOR_LIST[@]}"; do
        echo "====================================================="
        echo "    Starting Run: Factor=${factor}, Tokens=${MAX_EX_MB}M"
        echo "====================================================="

        # Dynamically set the expansion factor and run name for the current iteration
        EXPANSION_FACTOR=${factor}
        RUN_NAME="smollm2-sparsify-lmsys-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

        # The torchrun command is only written once
        torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
          --split "train[:${TOTAL_MAX}]" \
          --max_examples ${MAX_EX} \
          --expansion_factor ${EXPANSION_FACTOR} \
          --k ${K} \
          --batch_size ${BATCH_SIZE} \
          --grad_acc_steps ${GRAD_ACC_STEPS} \
          --ctx_len ${CTX_LEN} \
          --layers ${LAYERS} \
          --data_preprocessing_num_proc ${NUM_PROC} \
          --log_to_wandb \
          --run_name ${RUN_NAME} \
          --wandb_log_frequency ${WANDB_FREQ}

        echo "--- Finished Run: Factor=${factor}, Tokens=${MAX_EX_MB}M ---"
    done
done

echo "--- All runs completed. Job finished. ---"
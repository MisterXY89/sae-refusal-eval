#!/usr/bin/env bash

#SBATCH --job-name=smollm2-SAE
# alt: a100, a100s
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G   #64 for a100, 48 for a40
#SBATCH --time=10:00:00 
#SBATCH --output=logs/sparsify_smol2_%j.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END,FAIL

# Fail on error
set -e

# === Parameter Definitions ===
MODEL="HuggingFaceTB/SmolLM2-135M"
DATASET="MisterXY89/lmsys-processed"
LAYERS=18
NUM_PROC=4
# train tokens @ ctx_len=64
# ca 419M
# MAX_EX=6557946 # this is the max length supported by the dataset ~approx 419M
# ca 250M tokens
MAX_EX=3906200
MAX_EX_MB=$(( (MAX_EX * 64) / 1000000 ))

# === Hyperparameters ===
K=64
BATCH_SIZE=48   #32 for a40, 48 for A100
GRAD_ACC_STEPS=4
CTX_LEN=64

# === Logging ===
WANDB_FREQ=100

# === Environment setup ===
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

#######################################
#               RUN 1
#######################################
echo "--- Starting Run 1: Expansion Factor 16 ---"

# === Set parameters for Run 1 ===
EXPANSION_FACTOR=16
RUN_NAME="smollm2-sparsify-lmsys-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

# === Launch Run 1 ===
torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
  --split "train" \
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

echo "--- Run 1 finished. ---"

#######################################
#               RUN 2
#######################################
echo "--- Starting Run 2: Expansion Factor 32 ---"

# === Set parameters for Run 2 ===
EXPANSION_FACTOR=32
RUN_NAME="smollm2-sparsify-lmsys-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

# === Launch Run 2 ===
torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
  --split "train" \
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



#######################################
#               RUN 3 (MAX_EX CHANGE)/8 EXP
#######################################
echo ">>>> MAX_EX 125M <<<"
echo "--- Starting Run 3: Expansion Factor 8 ---"

# === Set parameters for Run 3 ===
MAX_EX=1953100
MAX_EX_MB=$(( (MAX_EX * 64) / 1000000 ))
EXPANSION_FACTOR=8
RUN_NAME="smollm2-sparsify-lmsys-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

# === Launch Run 1 ===
torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
  --split "train" \
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

echo "--- Run 3 finished. ---"


#######################################
#               RUN 4 (MAX_EX CHANGE)/16 EXP
#######################################
echo ">>>> MAX_EX 350M <<<"
echo "--- Starting Run 4: Expansion Factor 16 ---"

# === Set parameters for Run 3 ===
MAX_EX=1953100
MAX_EX_MB=$(( (MAX_EX * 64) / 1000000 ))
EXPANSION_FACTOR=16
RUN_NAME="smollm2-sparsify-lmsys-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

# === Launch Run 4 ===
torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
  --split "train" \
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

echo "--- Run 4 finished. ---"


#######################################
#               RUN 5 (MAX_EX CHANGE)/32 EXP
#######################################
echo ">>>> MAX_EX 350M <<<"
echo "--- Starting Run 5: Expansion Factor 32 ---"

# === Set parameters for Run 3 ===
MAX_EX=1953100
MAX_EX_MB=$(( (MAX_EX * 64) / 1000000 ))
EXPANSION_FACTOR=32
RUN_NAME="smollm2-sparsify-lmsys-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"

# === Launch Run 5 ===
torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE} -m sparsify ${MODEL} ${DATASET} \
  --split "train" \
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

echo "--- Run 5 finished. ---"



echo "--- Job finished. ---"
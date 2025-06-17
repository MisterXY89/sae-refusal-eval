#!/usr/bin/env bash

#SBATCH --job-name=exp-16-smollm2-sparsify
#SBATCH --partition=GPU-a100s
#SBATCH --nodes=1
# tasks refer to GPUs, so 1 node with 1 GPU, in case of 4 GPUs, --> 4 tasks
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sparsify_smol2_%j.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END,FAIL

# === Parameter Definitions ===
MODEL="HuggingFaceTB/SmolLM2-135M"
DATASET="monology/pile-uncopyrighted"

# Choosing 4 layers so distribute_modules (4 GPUs) can shard one SAE each
# LAYERS="12 18 24 29"
LAYERS=18
# # shard SAEs across GPUs --> only w
# DISTRIBUTE_MODULES=true

# Number of processes per node (for data preprocessing)
NUM_PROC=4
# approx: 500 M tokens @ ctx_len=64 (number of examples)
MAX_EX=6557946 # this is the max length supported by the dataset (LMSYS) ~approx 419M
# 1B tokens
# MAX_EX=15625000


# === Hyperparameters ===
# latent dim = d_model×8 = 576×8 = 4608
EXPANSION_FACTOR=8
# activate 64 of 4608 latents (∼1.4%)
K=64
# sequences per GPU (L(N) insensitive to batch size)
BATCH_SIZE=48
# accumulate over 4 steps
GRAD_ACC_STEPS=4
# micro-accumulation steps (for smaller GPUs, e.g. 24GB A40)
# MICRO_ACC_STEPS=1
# context length, as in Gao et al.
CTX_LEN=64
# save memory via 8-bit loading
# LOAD_IN_8BIT=true

# === Logging ===
MAX_EX_MB=$(( (MAX_EX * 64) / 1000000 ))
RUN_NAME="smollm2-sparsify-pile-${MAX_EX_MB}M-token-${LAYERS}-layers-${EXPANSION_FACTOR}-expansion-${K}-k"
WANDB_FREQ=100

# === Environment setup ===
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface



# === Launch sparsify via torchrun ===
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
  --wandb_log_frequency ${WANDB_FREQ} \
  # --loss_fn {ce,fvu,kl}
  # --micro_acc_steps ${MICRO_ACC_STEPS} \
  # --distribute_modules ${DISTRIBUTE_MODULES} \
  # --load_in_8bit ${LOAD_IN_8BIT} \ 

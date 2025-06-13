#!/usr/bin/env bash
#SBATCH --job-name=smollm2-sparsify            # job name
#SBATCH --partition=GPU-a40                   # partition with A40 GPUs
#SBATCH --nodes=1                             # single node
#SBATCH --ntasks-per-node=4                   # one task per GPU
#SBATCH --gres=gpu:a40:4                      # request 4 A40 GPUs
#SBATCH --cpus-per-task=4                     # CPUs per task
#SBATCH --mem=64G                             # total memory
#SBATCH --time=10:00:00                       # max runtime
#SBATCH --output=logs/sparsify_smol2_%j.out   # stdout log
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at  # notifications
#SBATCH --mail-type=END,FAIL

# === Parameter Definitions ===
MODEL="HuggingFaceTB/SmolLM2-135M"            # Hugging Face model identifier
# default dataset:
# DATASET="EleutherAI/SmolLM2-135M-10B"
DATASET="monology/pile-uncopyrighted"
SPLIT="train"                                # dataset split to use

# Choosing 4 layers so distribute_modules (4 GPUs) can shard one SAE each
LAYERS="12 18 24 29"                         # alt. LAYER_STRIDE
# DISTRIBUTE_MODULES=true                      # shard SAEs across GPUs

# Sparse‐AE hyperparameters
EXPANSION_FACTOR=8                           # latent dim = d_model×8 = 576×8 = 4608
K=64                                         # activate 64 of 4608 latents (∼1.4%)
BATCH_SIZE=8                                 # sequences per GPU (L(N) insensitive to batch size) :contentReference[oaicite:8]{index=8}
GRAD_ACC_STEPS=8                             # accumulate over 8 steps
MICRO_ACC_STEPS=2                            # split each batch into 2 microbatches
CTX_LEN=64                                   # context length, as in Gao et al. :contentReference[oaicite:9]{index=9}
LOAD_IN_8BIT=true                            # save memory via 8-bit loading

# Weights & Biases logging
LOG_TO_WANDB=true
RUN_NAME="smollm2-sparsify"
WANDB_FREQ=10

# === Environment setup ===
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

# === Build sparsify arguments ===
NUM_PROC=4
MAX_EX=7812500   # approx: 500 M tokens @ ctx_len=64


CMD_ARGS=(
  --expansion_factor ${EXPANSION_FACTOR}
  -k ${K}
  --batch_size ${BATCH_SIZE}
  --grad_acc_steps ${GRAD_ACC_STEPS}
  --micro_acc_steps ${MICRO_ACC_STEPS}
  --ctx_len ${CTX_LEN}
  --layers ${LAYERS}

  # limit HF preprocessors
  --data_preprocessing_num_proc ${NUM_PROC}
  # only scan a subset
  # --max_examples ${MAX_EX}

  --split "train[:${MAX_EX}]"
)

if [ "${DISTRIBUTE_MODULES}" = true ]; then
  CMD_ARGS+=(
    --distribute_modules
    --layer_stride ${LAYER_STRIDE}
  )
fi

if [ "${LOAD_IN_8BIT}" = true ]; then
  CMD_ARGS+=( --load_in_8bit )
fi

if [ "${LOG_TO_WANDB}" = true ]; then
  CMD_ARGS+=(
    --log_to_wandb
    --run_name ${RUN_NAME}
    --wandb_log_frequency ${WANDB_FREQ}
  )
fi

# === Launch sparsify via torchrun ===
torchrun --nproc_per_node=${SLURM_NTASKS_PER_NODE:-4} -m sparsify \
         ${MODEL} ${DATASET} "${CMD_ARGS[@]}"




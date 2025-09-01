#!/usr/bin/env bash
#SBATCH --job-name=eval-json-SAE
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/full_eval_%A_%a.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END
#SBATCH --array=0-4

set -e

# config
CONFIG_FILE="evaluation/steer_eval_config.json"
PRE_TRAINED_MODEL="HuggingFaceTB/SmolLM2-135M"

# env
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal
export HF_TOKEN=hf_XXXXXXX
export HF_HOME=/share/tilman.kerl/huggingface
cd /home/tilman.kerl/mech-interp/src

# task id → model name
SAE_INDEX=${SLURM_ARRAY_TASK_ID}
SPARSE_MODEL=$(jq -r ".[$SAE_INDEX].sae_id" $CONFIG_FILE)
echo "--- evaluations for SAE: $SPARSE_MODEL ---"

# if model name contains EleutherAI → treat as remote
if [[ "$SPARSE_MODEL" == *EleutherAI* ]]; then
    IS_LOCAL_FLAG="--no_local"
else
    IS_LOCAL_FLAG="--is_local"
fi

# iterate layers
NUM_LAYERS=$(jq ".[$SAE_INDEX].layers | length" $CONFIG_FILE)
for (( i=0; i<$NUM_LAYERS; i++ )); do
    HOOKPOINT=$(jq -r ".[$SAE_INDEX].layers[$i].hookpoint" $CONFIG_FILE)
    FEATURE_INDEX_LIST=$(jq -r ".[$SAE_INDEX].layers[$i].indices" $CONFIG_FILE)
    STEERING_COEF_LIST=$(jq -r ".[$SAE_INDEX].layers[$i].coeffs" $CONFIG_FILE)

    echo "--- layer: $HOOKPOINT ---"
    RUN_TAG=$(basename "${SPARSE_MODEL%/}")
    STEER_PATH="steer_configs/${RUN_TAG}-${HOOKPOINT}.pt"
    mkdir -p "$(dirname "$STEER_PATH")"

    python -m evaluation.run_eval \
      --model_type steered \
      --pretrained "$PRE_TRAINED_MODEL" \
      --steer_path "$STEER_PATH" \
      --loader sparsify \
      --action add \
      --sparse_model "$SPARSE_MODEL" \
      $IS_LOCAL_FLAG \
      --hookpoint "$HOOKPOINT" \
      --feature_indices $FEATURE_INDEX_LIST \
      --steering_coefficients $STEERING_COEF_LIST \
      --tasks mmlu,hellaswag \
      --batch_size 64 \
      --device cuda:0

    echo "--- done: $HOOKPOINT ---"
done

echo "--- all evaluations for $SPARSE_MODEL done ---"

#!/usr/bin/env bash
#SBATCH --job-name=full-delphi-smollm2-SAEs
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/delphi_eval_%j.out
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END

# "/home/tilman.kerl/mech-interp/src/train/MIX/checkpoints/smollm2-sparsify-EQ-50M-token-6_25-layers-32-expansion-64-k"
  # "/home/tilman.kerl/mech-interp/src/train/LMSYS/checkpoints/smollm2-sparsify-lmsys-50M-token-6_25-layers-32-expansion-64-k"
  # "/home/tilman.kerl/mech-interp/src/train/MIX/checkpoints/smollm2-sparsify-INS-50M-token-6_25-layers-32-expansion-64-k"
# four SAE checkpoints
sae_models=(  
  "/home/tilman.kerl/mech-interp/src/train/MIX/checkpoints/smollm2-sparsify-PRE-50M-token-6_25-layers-32-expansion-64-k"
)

# two layer settings
layers=(25)

# fixed pretrained model
pre_trained_model="HuggingFaceTB/SmolLM2-135M"

# activate conda and env
source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh
conda activate refusal

export HF_TOKEN=hf_XXXXXXX
export HF_HOME=/share/tilman.kerl/huggingface
export OPENROUTER_API_KEY=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

# optionally disable HF-datasets multiprocessing entirely
# export HF_DATASETS_DISABLE_MULTIPROCESSING=1

for sparse_model in "${sae_models[@]}"; do
  for layer in "${layers[@]}"; do
    run_name="EXP-$(basename "$sparse_model")-layer-$layer"
    echo "=== Running ${run_name} ==="

    python -m delphi \
      "$pre_trained_model" \
      "$sparse_model" \
      --hookpoints "layers.$layer" \
      --scorers detection \
      --filter_bos \
      --name "$run_name" \
      --dataset_repo MisterXY89/SmolLM-lmsys-mixtures \
      --dataset_split balanced_50_50[:1%] \
      --load_in_8bit true \
      --cache_ctx_len 128 \
      --dataset_column text \
      --hf_token "$HF_TOKEN" \
      --num_gpus 1 \
      --max_latents 1000
      # --explainer_provider openrouter \
      # --explainer_model google/gemini-2.5-flash
  done
done

#!/bin/bash
#SBATCH --job-name=lmsys-sae-pythia-train
#SBATCH --partition=GPU-a40
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/lmsys_sae_train_%j.out
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END,FAIL

source /home/tilman.kerl/miniconda3/etc/profile.d/conda.sh 
conda activate refusal

export HF_TOKEN=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
export HF_HOME=/share/tilman.kerl/huggingface

python lmsys_train.py

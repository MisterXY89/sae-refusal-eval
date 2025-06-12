#!/bin/bash
#SBATCH --job-name=pythia-sk-train
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=logs/sae_train_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=tilman.kerl@tuwien.ac.at
#SBATCH --mail-type=END,FAIL

conda activate refusal
export HF_TOKEN=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
python train.py
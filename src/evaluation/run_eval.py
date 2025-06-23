#!/usr/bin/env python3

import sys
import argparse
import subprocess
import pandas as pd

import json
from tqdm import tqdm

import lm_eval
from lm_eval.utils import setup_logging

from steering.sae.sparsify import (
    make_steered_hf_lm, generate_with_steered_hf
)
from utils.generation import HF_GENERATION_KW_ARGS
from data_tools.instructions import (
    get_harmful_instructions, get_harmless_instructions
)
from evaluation.refusal import (
    get_refusal_scores, get_wildguard_refusal_score, get_semantic_refusal_score
)

TEST_SIZE = 100
harmless_inst_train, harmless_inst_test = get_harmless_instructions()
harmful_inst_train, harmful_inst_test = get_harmful_instructions()

def refusal_eval(args):
    steer_cfg = {
        args.hookpoint: {
            "action": args.action,
            "sparse_model": args.sparse_model,
            "feature_index": args.feature_index,
            "steering_coefficient": args.steering_coefficient,
            "loader": args.loader,
            "sae_id": "",
            "description": f"steering feature {args.feature_index}",
        }
    }
    
    hf_lm = make_steered_hf_lm(
        steer_cfg,
        pretrained=args.pretrained,
        device="cuda:0",
        batch_size=32,
        # can be overwritten
        # gen_kwargs={},
        seed=42,
    )

    steered_generation_harmful = [
        generate_with_steered_hf(hf_lm, harmful_inst) 
        for harmful_inst in tqdm(harmful_inst_test[:TEST_SIZE])
    ]
    rr_2 = get_refusal_scores(steered_generation_harmful)
    # rr = get_wildguard_refusal_score(harmful_inst_test[:TEST_SIZE], steered_generation_harmful)
    rr = get_semantic_refusal_score(steered_generation_harmful)

    # print(steered_generation_harmful)
    
    steered_generation_harmless = [
        generate_with_steered_hf(hf_lm, harmless_inst) 
        for harmless_inst in tqdm(harmless_inst_test[:TEST_SIZE])
    ]

    # print(steered_generation_harmless)
    orr_2 = get_refusal_scores(steered_generation_harmless)
    # orr = get_wildguard_refusal_score(harmless_inst_test[:TEST_SIZE], steered_generation_harmless)
    orr = get_semantic_refusal_score(steered_generation_harmless)

    results = {
        "steer_cfg": steer_cfg,
        "rr": rr,
        # "rr_2": rr_2,
        "orr": orr,
        # "orr_2": orr_2
    }

    print(results)

    sae_name = args.sparse_model.split("/")[1]
    with open(f"/home/tilman.kerl/mech-interp/src/results/perfomance/refusal/{sae_name}.json", "w") as f:
        json.dump(results, f, indent=2)
        
    

def main():
    p = argparse.ArgumentParser(
        description="Run LM-Eval with optional SAE steering"
    )
    p.add_argument(
        "--model_type",
        choices=["hf", "steered"],
        required=True,
        help="hf = vanilla HF model, steered = apply SAE steering"
    )
    p.add_argument(
        "--pretrained",
        required=True,
        help="HuggingFace model name (e.g. EleutherAI/pythia-410m-deduped)"
    )
    p.add_argument(
        "--steer_path",
        default="steer_config.csv",
        help="path to steering CSV (only used if model_type=steered)"
    )
    # individual steering params, if you prefer to generate the CSV here:
    p.add_argument("--loader", help="steering loader (e.g. sparsify)")
    p.add_argument("--action", help="steering action (add|mul|etc.)")
    p.add_argument(
        "--sparse_model",
        help="SAE repo (e.g. EleutherAI/sae-pythia-410m-65k)"
    )
    p.add_argument("--hookpoint", help="hookpoint name (e.g. layers.5.mlp)")
    p.add_argument("--feature_index", type=int, help="latent feature index")
    p.add_argument(
        "--steering_coefficient", type=float,
        help="strength (positive or negative) to add/mul"
    )

    p.add_argument(
        "--tasks",
        default="mmlu,realtoxicityprompts,toxigen,hendrycks_ethics",
        help="comma-separated list of tasks"
    )
    p.add_argument("--device", default="cuda:0", help="device for generation")
    p.add_argument(
        "--wandb_project",
        default="MA-sae-eval",
        help="WandB project name (optional)"
    )
    p.add_argument("--limit", type=int, help="max examples per task")
    p.add_argument("--batch_size", type=int, help="batch size for LM-Eval")

    args = p.parse_args()

    
    # If steered and user provided individual params, build the CSV:
    if args.model_type == "steered" and args.loader and args.action:
        df = pd.DataFrame({
            "loader": [args.loader],
            "action": [args.action],
            "sparse_model": [args.sparse_model],
            "hookpoint": [args.hookpoint],
            "feature_index": [args.feature_index],
            "steering_coefficient": [args.steering_coefficient],
        })
        df.to_csv(args.steer_path, index=False)
        print(f"→ Wrote steering CSV to {args.steer_path}")

    # Build model_args string
    if args.model_type == "steered":
        model_args = f"pretrained={args.pretrained},steer_path={args.steer_path}"
    else:
        model_args = f"pretrained={args.pretrained}"

    sae_name = args.sparse_model.split("/")[1]
    
    output_path = f"/home/tilman.kerl/mech-interp/src/results/perfomance/downstream/{sae_name}.json"

    # Assemble the lm_eval command
    cmd = [
        "lm_eval",
        "--model", args.model_type,
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--device", args.device,
        "--output_path", output_path,
        # "--wandb_args", f"project={args.wandb_project}",
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]

    print("→ Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)

    #now we can also run the rr & orr eval:
    refusal_eval(args)
    
    

if __name__ == "__main__":
    """
1) Vanilla HF eval
python run_eval.py \
  --model_type hf \
  --pretrained EleutherAI/pythia-410m-deduped \
  --tasks mmlu,realtoxicityprompts,toxigen,hendrycks_ethics \
  --device cuda:0 \
  --wandb_project MA-sae-eval \
  --limit 10 \
  --batch_size 8


2) Steered eval with existing CSV
python run_eval.py \
  --model_type steered \
  --pretrained EleutherAI/pythia-410m-deduped \
  --steer_path my_steer_config.csv \
  --tasks mmlu,realtoxicityprompts,toxigen,hendrycks_ethics \
  --device cuda:0 \
  --wandb_project MA-sae-eval \
  --limit 10 \
  --batch_size 8
  
3) Steered eval by specifying steering params
python run_eval.py \
  --model_type steered \
  --pretrained EleutherAI/pythia-410m-deduped \
  --loader sparsify \
  --action add \
  --sparse_model EleutherAI/sae-pythia-410m-65k \
  --hookpoint layers.5.mlp \
  --feature_index 17 \
  --steering_coefficient 10.0 \
  --tasks mmlu,realtoxicityprompts,toxigen,hendrycks_ethics \
  --device cuda:0 \
  --wandb_project MA-sae-eval \
  --limit 10 \
  --batch_size 8
    """
    main()
"""
python run_eval.py \
  --model_type steered \
  --pretrained HuggingFaceTB/SmolLM2-135M \
  --loader sparsify \
  --action add \
  --sparse_model /home/tilman.kerl/mech-interp/src/train/LMSYS/checkpoints/smollm2-sparsify-lmsys-419M-token-18-layers-16-expansion-64-k/ \
  --hookpoint layers.18 \
  --feature_index 17 \
  --steering_coefficient 10.0 \
  --tasks mmlu \
  --device cuda:0 \
  --wandb_project MA-sae-eval \
  --limit 10 \
  --batch_size 8
    
"""
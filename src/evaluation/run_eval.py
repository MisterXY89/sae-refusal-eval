
import sys
import argparse
import subprocess
import pandas as pd

import json
from tqdm import tqdm

from sparsify import Sae

import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.models.hf_steered import SteeredModel

from steering.sae.sparsify import (
#    make_steered_hf_lm, 
    generate_with_steered_hf
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

def _load_saes(layers, sae_path: str, is_local: bool = True):
    # load SAE models for each layer, either from disk or hub
    get_hookpoint = lambda l: f"layers.{l}" if is_local else f"layers.{l}.mlp"
    saes = {
        layer: (
            Sae.load_from_disk(f"{sae_path}/{get_hookpoint(layer)}", device="cuda:0")
            if is_local
            else Sae.load_from_hub(sae_path, hookpoint=get_hookpoint(layer), device="cuda:0")
        )
        for layer in layers
    }
    return saes


def refusal_eval(args):
    import torch
    
    # extract integer layer index from hookpoint, e.g. "layers.20.mlp" → 20
    layer = int(args.hookpoint.split(".")[1])
    saes = _load_saes([layer], args.sparse_model, is_local=args.is_local)
    coder = saes[layer]    

    # build combined steering vector
    vecs = [
        coef * coder.W_dec[idx]
        for idx, coef in zip(args.feature_indices, args.steering_coefficients)
    ]
    combined = torch.stack(vecs, dim=0).sum(dim=0)
    bias = coder.b_dec

    # assemble and save .pt
    steer_dict = {
        args.hookpoint: {
            "action": args.action,
            "steering_vector": combined,
            "steering_coefficient": 1.0,
            "bias": bias,
        }
    }    
    torch.save(steer_dict, args.steer_path)
    print(f"Saved combined steering vector to {args.steer_path}")

    # instantiate steered model
    hf_lm = SteeredModel(
        pretrained=args.pretrained,
        steer_path=args.steer_path,
        device=args.device        
    )

    # harmful
    steered_harm = [
        generate_with_steered_hf(hf_lm, inst)
        for inst in tqdm(harmful_inst_test[:TEST_SIZE], desc="Harmful")
    ]
    rr_2 = get_refusal_scores(steered_harm)
    rr = get_semantic_refusal_score(steered_harm)

    print("\nSample harmful:")
    for gen in steered_harm[:5]:
        print(gen)

    # harmless
    steered_harmless = [
        generate_with_steered_hf(hf_lm, inst)
        for inst in tqdm(harmless_inst_test[:TEST_SIZE], desc="Harmless")
    ]
    orr_2 = get_refusal_scores(steered_harmless)
    orr = get_semantic_refusal_score(steered_harmless)

    print("\nSample harmless:")
    for gen in steered_harmless[:5]:
        print(gen)

    # results
    results = {
        "steer_path": args.steer_path,
        "feature_indices": args.feature_indices,
        "steering_coefficients": args.steering_coefficients,
        "rr": rr,
        "rr_2": rr_2,
        "orr": orr,
        "orr_2": orr_2,
    }
    print("\nFinal results:", json.dumps(results, indent=2))
    sae_name = args.sparse_model.rstrip("/").split("/")[-1] 
    out_path = f"/home/tilman.kerl/mech-interp/src/results/perfomance/refusal/{sae_name}-{args.hookpoint}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
        
    

def main():
    p = argparse.ArgumentParser(description="Run LM-Eval with optional SAE steering")
    p.add_argument(
        "--model_type",
        choices=["hf", "steered"],
        required=True,
        help="hf = vanilla HF, steered = apply SAE steering",
    )
    p.add_argument(
        "--pretrained",
        required=True,
        help="HuggingFace model (e.g. EleutherAI/pythia-410m-deduped)",
    )
    p.add_argument(
        "--steer_path",
        default="steer_config.pt",
        help="output .pt for steering vectors (used if model_type=steered)",
    )
    p.add_argument(
        "--loader",
        required=True,
        choices=["sparsify", "sae_lens"],
        help="steering loader",
    )
    p.add_argument(
        "--action", required=True, choices=["add", "clamp"], help="steering action"
    )
    p.add_argument(
        "--sparse_model",
        required=True,
        help="SAE repo or local path (e.g. EleutherAI/sae-pythia-410m-65k)",
    )
    p.add_argument(
        "--hookpoint",
        required=True,
        help="hookpoint name (e.g. layers.5 or layers.5.mlp)",
    )
    p.add_argument(
        "--feature_indices",
        type=int,
        nargs="+",
        required=True,
        help="one or more feature indices",
    )
    p.add_argument(
        "--steering_coefficients",
        type=float,
        nargs="+",
        required=True,
        help="matching strengths for each feature",
    )
    p.add_argument(
        "--is_local",
        dest="is_local",
        action="store_true",
        help="load SAEs from disk (default)",
    )
    p.add_argument(
        "--no_local",
        dest="is_local",
        action="store_false",
        help="load SAEs from hub",
    )
    p.set_defaults(is_local=True)

    p.add_argument(
        "--tasks",
        default="mmlu,realtoxicityprompts,toxigen,hendrycks_ethics",
        help="comma-separated list of tasks",
    )
    p.add_argument(
        "--wandb_project",
        default="MA-sae-eval",
        help="WandB project name (optional)"
    )
    p.add_argument("--limit", type=int, help="max examples per task")
    p.add_argument("--device", default="cuda:0", help="generation device")
    p.add_argument("--batch_size", type=int, default=32, help="LM-Eval batch size")

    args = p.parse_args()

    refusal_eval(args)

    
    ## build the model_args string to pass to lm_eval
    if args.model_type == "steered":
        model_args = f"pretrained={args.pretrained},steer_path={args.steer_path}"
    else:
        model_args = f"pretrained={args.pretrained}"

    sae_name = args.sparse_model.rstrip("/").split("/")[-1]
    # path is a dir
    output_path = f"/home/tilman.kerl/mech-interp/src/results/perfomance/downstream/{sae_name}-{args.hookpoint}"

    # assemble the lm_eval command
    cmd = [
        "lm_eval",
        "--model", args.model_type,
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--device", args.device,
        "--output_path", output_path,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.batch_size:
        cmd += ["--batch_size", str(args.batch_size)]

    print("→ Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    
    

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
  --batch_size 64


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
#!/usr/bin/env python3

import sys
import argparse
import subprocess
import pandas as pd

import pathlib
from pathlib import Path

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

harmless_inst_train, harmless_inst_test = get_harmless_instructions()
harmful_inst_train, harmful_inst_test = get_harmful_instructions()

def refusal_eval(
    sparse_model: str, 
    feature_index: int, 
    hookpoint: str,
    pretrained: str = "HuggingFaceTB/SmolLM2-135M" , 
    action: str = "add", 
    steering_coefficient: float = 3, 
    test_size = 10
):
    steer_cfg = {
        hookpoint: {
            "action": action,
            "sparse_model": sparse_model,
            "feature_index": feature_index,
            "steering_coefficient": steering_coefficient,
            "loader": "sparsify", 
            "sae_id": "",
            "description": f"steering feature {feature_index}",
        }
    }
    
    hf_lm = make_steered_hf_lm(
        steer_cfg,
        pretrained=pretrained,
        device="cuda:0",
        batch_size=32,
        seed=42, # gen_kwargs={},        
    )

    steered_generation_harmful = [
        generate_with_steered_hf(hf_lm, harmful_inst) 
        for harmful_inst in harmful_inst_test[:test_size]
    ]
    rr_2 = get_refusal_scores(steered_generation_harmful)
    rr = get_semantic_refusal_score(steered_generation_harmful)
    
    steered_generation_harmless = [
        generate_with_steered_hf(hf_lm, harmless_inst) 
        for harmless_inst in harmless_inst_test[:test_size]
    ]

    orr_2 = get_refusal_scores(steered_generation_harmless)
    orr = get_semantic_refusal_score(steered_generation_harmless)

    tmp_results = {
        "name": sparse_model.split("/")[1],
        "steer_cfg": steer_cfg,
        "rr": rr,
        "rr_2": rr_2,
        "orr": orr,
        "orr_2": orr_2
    }

    # print(tmp_results)
    return tmp_results


def grid_search(sparse_model, pretrained, layer, exp_factor: int = 16):
    base_size = 576
    total_hidden_size = base_size * exp_factor
    features = list(range(0, total_hidden_size, 100))

    hookpoint = f"layers.{layer}"
    if "EleutherAI" in sparse_model:
        hookpoint += ".mlp"

    search_results = {}
    for feat_idx in tqdm(features):
        feat_results = refusal_eval(
            sparse_model=sparse_model,
            pretrained=pretrained,
            feature_index=feat_idx,
            hookpoint=hookpoint,
        )
        search_results[feat_idx] = feat_results

    sae_name = pathlib.Path(sparse_model).name
    out_path = f"/home/tilman.kerl/mech-interp/src/results/saes/features/grid_{sae_name}.json"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(search_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid-search refusal-rate impact of individual SAE features."
    )
    parser.add_argument(
        "--sparse_model",
        required=True,
        help="HF repo or local path of the sparse-autoencoder checkpoint",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="HF repo or local path of the pre-trained checkpoint",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Transformer layer index at which the SAE is hooked.",
    )
    parser.add_argument(
        "--exp_factor",
        type=int,
        default=16,
        help="Expansion factor used when the SAE was trained (default: 16).",
    )
    args = parser.parse_args()

    grid_search(args.sparse_model, args.pretrained, args.layer, args.exp_factor)

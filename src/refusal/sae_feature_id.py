import os
import json
import pathlib
import numpy as np
import pandas as pd
from sparsify import Sae


from data_tools.instructions import get_harmful_instructions, get_harmless_instructions
from utils.templates import PYTHIA_TEMPLATE
from utils.generation import ( 
    format_instruction, tokenize_instructions
)
from utils.common import (
    _to_jsonable, convert_dfs_to_json_serializable
)

import steering.linear_probing as lp_steer
import refusal.linear_probing as lp_refuse

from refusal.sae.sparsify.latent_features import get_latent_feature_stats as sparsify_get_latent_feature_stats
from refusal.sae.sparsify import utils as sparsify_utils

from evaluation.refusal import (
    get_refusal_scores, get_wildguard_refusal_score
)
from evaluation.sae_eval import compute_sae_stats


harmless_inst_train, harmless_inst_test = get_harmless_instructions()
harmful_inst_train, harmful_inst_test = get_harmful_instructions()

LOCAL_LAYERS = [6, 25]
HUB_LAYERS = list(range(0,30,3))

def _build_ds(n_inst_train: int):
    return (
        [{"prompt": p, "label": 0} for p in harmless_inst_train[:n_inst_train]] +
        [{"prompt": p, "label": 1} for p in harmful_inst_train[:n_inst_train]]
    )


def id_refusal_feature_for_sae(
    sae_config: dict,
    top_N: int = 10,
    n_inst_train: int = 100,
    layers = None,  # uses default
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    visualise_top_features: bool = True,
    visualise_latent_difference: bool = False,
    save: bool = True,
    rank_metric: str = "cohens_d", # "fisher"
):
    if not sae_config["path"]: 
        return 0

    sae_name = f"{sae_config['name']}-layer-{sae_config['layer']}"

    dataset = _build_ds(n_inst_train=n_inst_train)

    if not layers:    
        layers = LOCAL_LAYERS if sae_config["is_local"] else HUB_LAYERS
        
    results, harmful_reps_list, harmless_reps_list, diff_list = compute_sae_stats(
        sae_config["path"],
        model_name, 
        layers, 
        dataset = dataset,
        exp_factor = sae_config["exp_factor"],
        max_length = sae_config["max_length"],
        is_local = sae_config["is_local"],
    )

    cohens_d, fisher = sparsify_utils.compute_effect_sizes(results, layer=0)
    results[0]['stats']['cohens_d'] = cohens_d
    results[0]['stats']['fisher'] = fisher

    feature_summary = sparsify_utils.identify_top_features(
        results.copy(),
        layers=layers, 
        N=top_N,
        # metric="cohens_d",
        rank_metric=rank_metric,
    )

    if visualise_latent_difference:
        sparsify_utils.visualize_latent_differences(
            harmful_reps_list, harmless_reps_list, diff_list,
            sae_name
        )
    if visualise_top_features:
        harmful_reps = harmful_reps_list[0]
        harmless_reps = harmless_reps_list[0]
        
        # Call the new all-in-one visualization function
        sparsify_utils.create_sae_feature_dashboard(
            full_results_obj=results,
            harmful_reps=harmful_reps,
            harmless_reps=harmless_reps,
            feature_summary=feature_summary,
            sae_name=sae_name
        )

    print("Top features overall (harmful):")
    display(feature_summary["top_harmful"].head())
    print("Top features overall (harmless):")
    display(feature_summary["top_harmless"].head())

    if save:    
        payload = {
            # "results": _to_jsonable(convert_dfs_to_json_serializable(results)),
            "summary": _to_jsonable(convert_dfs_to_json_serializable(feature_summary.copy())),
        }    
        # construct ./results/saes/features/{SAE_path}.json    
        out_dir = pathlib.Path("./results/saes/features")
        out_dir.mkdir(parents=True, exist_ok=True)
        outfile = out_dir / f"{sae_name}.json"
    
        with outfile.open("w") as fp:
            json.dump(payload, fp, indent=2)

    return feature_summary


        
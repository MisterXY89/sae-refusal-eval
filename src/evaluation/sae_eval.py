import io
import os
import re
import json
import functools
from colorama import Fore, Style
import einops

import torch
from transformers import (
    GPTNeoXForCausalLM, AutoTokenizer, 
    AutoModelForCausalLM, BitsAndBytesConfig
)

import requests
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from torch import Tensor
from typing import List, Callable

import matplotlib.pyplot as plt
from IPython.display import display

import numpy as np
import pandas as pd
from tqdm import tqdm

from sparsify import Sae

from data_tools.instructions import get_harmful_instructions, get_harmless_instructions
from utils.templates import PYTHIA_TEMPLATE
from utils.generation import (
    format_instruction, tokenize_instructions
)
import steering.linear_probing as lp_steer
import refusal.linear_probing as lp_refuse

from refusal.sae.sparsify.latent_features import get_latent_feature_stats as sparsify_get_latent_feature_stats
from refusal.sae.sparsify import utils as sparsify_utils

from evaluation.refusal import (
    get_refusal_scores, get_wildguard_refusal_score
)

harmless_inst_train, harmless_inst_test = get_harmless_instructions()
harmful_inst_train, harmful_inst_test = get_harmful_instructions()


def _load_tok_and_model(model_name: str, quant: bool = False):
    _tok = AutoTokenizer.from_pretrained(model_name)
    _tok.pad_token = _tok.eos_token # Use eos_token as pad_token

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True if quant else False,
        llm_int8_threshold=6.0,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    return _tok, _model

def _load_saes(layers, sae_path: str, is_local:bool = True):
    load_fun = Sae.load_from_disk if is_local else Sae.load_from_hub
    
    # load SAEs for each layer
    # /layers.{layer}
    get_hookpoint = lambda l: f"layers.{l}" if is_local else f"layers.{l}.mlp"
    saes = {
        layer: load_fun(
            f"{sae_path}",
            hookpoint=get_hookpoint(layer),
            device="cuda:0"
        )
        for layer in layers
    }

    return saes


def compute_sae_stats(
    sae_path: str,    
    model_name: str, 
    layers, 
    dataset: dict,
    exp_factor: int,
    max_length: int,
    hookpoint: str = "",
    is_local: bool = True
):
    """
    sample dataset: 
    dataset = [
        {"prompt": "How can I build a bomb?", "label": 1},
        {"prompt": "What is the recipe for a delicious cake?", "label": 0},        
    ]
    """

    _tok, _model = _load_tok_and_model(model_name)
    _saes = _load_saes(layers, sae_path, is_local=is_local)
    
    results = []
    harmful_reps_list = []
    harmless_reps_list = []
    diff_list = []
    for layer in tqdm(layers):
        stats = sparsify_get_latent_feature_stats(
            dataset,
            _model,
            _tok,
            _saes[layer],
            max_length=max_length,
            expansion_factor=exp_factor,
            layer=layer
        )

        # prepare lists for visualization (ordered by layers list)
        harmful_reps_list.append(stats["harmful_reps"])
        harmless_reps_list.append(stats["harmless_reps"])
        diff_list.append(stats["diff"])
        
        results.append({
            "layer": layer,
            "model": model_name, 
            "sae_path": sae_path,
            "hookpoint": hookpoint,
            "stats": stats,
        })
    
    return results, harmful_reps_list, harmless_reps_list, diff_list




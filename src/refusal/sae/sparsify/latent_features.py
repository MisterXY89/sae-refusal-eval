import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sparsify import Sae
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .representations import get_prompt_representation


def get_latent_feature_stats(
    dataset, model, tokenizer, sae, layer, max_length=2048, expansion_factor=32
):
    """
    Process a dataset of prompts (each a dict with keys "prompt" and "label" where label 1 = harmful, 0 = harmless),
    extract aggregated SAE representations, and compute the mean activation per latent dimension for each group.

    Args:
        dataset: A list of dictionaries, where each dictionary has keys "prompt" and "label".
        model: The language model.
        tokenizer: The tokenizer.
        sae: The sparse autoencoder.
        layer (int): The layer number to extract activations from.
        max_length: The maximum length of the prompt.
        expansion_factor: The expansion factor of the SAE.

    Returns a dict with:
      all_reps: [num_prompts, latent_dim] numpy array of representations.
      labels: numpy array of labels.
      harmful_reps: representations for harmful prompts.
      harmless_reps: representations for harmless prompts.
      mean_harmful: mean vector for harmful prompts.
      mean_harmless: mean vector for harmless prompts.
      diff: mean_harmful - mean_harmless.
    """
    all_reps = []
    labels = []

    for entry in tqdm(dataset, desc="Extracting representations"):
        rep, seq_len = get_prompt_representation(
            entry["prompt"],
            model,
            tokenizer,
            sae,
            layer=layer,
            max_length=max_length,
            expansion_factor=expansion_factor,
        )
        all_reps.append(rep.cpu().numpy())
        labels.append(entry["label"])

    all_reps = np.stack(all_reps, axis=0)
    labels = np.array(labels)

    harmful_reps = all_reps[labels == 1]
    harmless_reps = all_reps[labels == 0]

    mean_harmful = harmful_reps.mean(axis=0)
    mean_harmless = harmless_reps.mean(axis=0)
    diff = mean_harmful - mean_harmless

    return {
        "all_reps": all_reps,
        "labels": labels,
        "harmful_reps": harmful_reps,
        "harmless_reps": harmless_reps,
        "mean_harmful": mean_harmful,
        "mean_harmless": mean_harmless,
        "diff": diff,
    }


def sort_and_flatten_stats(stats_list):
    """
    Takes a list of dictionaries (each containing stats for a layer),
    computes the difference for each latent dimension across all layers,
    flattens the results, and returns a sorted DataFrame.

    Args:
        stats_list: A list of dictionaries, where each dictionary is the output of get_latent_feature_stats.

    Returns:
        A Pandas DataFrame sorted by the difference across all layers.
    """
    all_diffs = []
    for i, stats in enumerate(stats_list):
        diff = stats["diff"]
        layer_num = i  # Assuming the order of stats_list corresponds to layer number
        # Append layer number to each diff to keep track of which layer it came from
        all_diffs.extend([(layer_num, dim, diff_val) for dim, diff_val in enumerate(diff)])

    # Sort by difference
    sorted_diffs = sorted(all_diffs, key=lambda x: x[2], reverse=True)

    # Create a DataFrame
    df = pd.DataFrame(sorted_diffs, columns=["Layer", "Latent Dimension", "Difference"])

    return df

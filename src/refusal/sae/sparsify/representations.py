import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sparsify import Sae
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_prompt_representation(
    prompt, model, tokenizer, sae, layer, max_length=2048, expansion_factor=32
):
    """
    Tokenize a prompt with padding to max_length, run the model and SAE to extract activations from the specified layer,
    then aggregate the SAE top activations over the candidate dimension and token positions.

    Args:
        prompt (str): The input prompt.
        model (torch.nn.Module): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        sae (Sae): The sparse autoencoder.
        layer (int): The layer number to extract activations from.
        max_length (int, optional): The maximum sequence length. Defaults to 2048.
        expansion_factor (int, optional): The expansion factor of the SAE. Defaults to 32.

    Returns:
      rep: a 1D tensor of shape [latent_dim] representing the aggregated prompt representation.
      actual_seq_length: number of real (non-padded) tokens.
    """
    # Tokenize with padding to the SAE’s expected length.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    actual_seq_length = int(inputs["attention_mask"].sum().item())

    # Move inputs to model's device.
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        # Assume embeddings are at index 0
        hidden_state_layer = outputs.hidden_states[
            layer + 1
        ]  # +1 because embeddings are at index 0
        latent_features = sae.encode(hidden_state_layer)

    # latent_features.top_acts has shape [1, latent_dim, flat_length] where flat_length = max_length * expansion_factor.
    batch_size, latent_dim, flat_length = latent_features.top_acts.shape
    valid_flat_length = actual_seq_length * expansion_factor

    # Slice only the valid (non-padded) portion.
    top_acts_valid = latent_features.top_acts[:, :, :valid_flat_length]
    # Reshape to [batch, latent_dim, actual_seq_length, expansion_factor]
    reshaped_acts = top_acts_valid.view(
        batch_size, latent_dim, actual_seq_length, expansion_factor
    )

    # For each token position, take the max over the candidate (expansion) dimension,
    # then average over tokens to yield a vector of shape [latent_dim].
    rep = reshaped_acts.max(dim=-1)[0].mean(dim=-1)
    return rep.squeeze(0), actual_seq_length

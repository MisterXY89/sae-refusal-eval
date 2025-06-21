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
    # 1) tokenize & pad
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    actual_seq_length = int(inputs["attention_mask"].sum().item())
    # print(actual_seq_length)
    # print(expansion_factor)

    # 2) move to device & run
    device = next(model.parameters()).device
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        hidden = outputs.hidden_states[layer + 1]  # +1 since hidden_states[0] is embeddings
        latent = sae.encode(hidden)

    per_token_acts = latent.top_acts  # e.g., shape [1, 32768, 32]
    
    # Add a print statement to verify the shape during a run
    # print("Shape of latent.top_acts:", per_token_acts.shape)
    # print("Actual sequence length:", actual_seq_length)

    # 4) Aggregate: We no longer need to create an `expansion_factor` dimension.
    #    The goal is to get a representation of shape [F]. We can do this
    #    by taking the mean across the sequence length dimension.
    
    # Ensure you only average over the real tokens, not padding.
    # The SAE output is likely already unpadded, so its seq_len dim should equal actual_seq_length.
    rep = per_token_acts.mean(dim=-1)  # Take mean over the sequence_length dimension -> shape [B, F]

    return rep.squeeze(0), actual_seq_length



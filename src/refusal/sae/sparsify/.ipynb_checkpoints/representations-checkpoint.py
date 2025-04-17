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

    # 2) move to device & run
    device = next(model.parameters()).device
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        hidden = outputs.hidden_states[layer + 1]  # +1 since hidden_states[0] is embeddings
        latent = sae.encode(hidden)

    # 3) unpack shapes
    B, F, flat_length = latent.top_acts.shape
    seq_length = flat_length // expansion_factor  # e.g., 65536//32 = 2048

    # 4) reshape into [B, F, seq_length, expansion_factor]
    acts = latent.top_acts.view(B, F, seq_length, expansion_factor)

    # 5) mask out padded token positions:
    # build a mask of shape [seq_length] where True for padded positions
    device = acts.device
    token_idx = torch.arange(seq_length, device=device)
    mask = token_idx[None, None, :] >= actual_seq_length  # [1,1,seq_length]
    # extend mask to [B,F,seq_length,1] and fill with -inf
    acts = acts.masked_fill(mask.unsqueeze(-1), float('-inf'))

    # 6) aggregate: max over expansion dimension -> [B, F, seq_length]
    per_token_max = acts.max(dim=-1)[0]
    # then mean over only the first actual_seq_length positions
    rep = per_token_max[:, :, :actual_seq_length].mean(dim=-1)  # [B, F]

    return rep.squeeze(0), actual_seq_length
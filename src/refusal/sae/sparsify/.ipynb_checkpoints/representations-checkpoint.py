mport matplotlib.pyplot as plt
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
    # 1) Tokenize & pad
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    actual_seq_length = int(inputs["attention_mask"].sum().item())

    # 2) Move to device & run model
    device = next(model.parameters()).device
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        # hidden has shape [batch_size=1, seq_len, hidden_dim]
        hidden = outputs.hidden_states[layer + 1]

        # 3) Flatten hidden state for the SAE and encode
        # Shape changes: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
        hidden_2d = hidden.squeeze(0)
        
        # Now pass the 2D tensor to the SAE
        latent = sae.encode(hidden_2d)

    # 4) Get the full pre-activation vector for the last non-padded token
    # `latent.pre_acts` has shape [seq_len, num_features]
    # We index at `actual_seq_length - 1` to get the last token's vector.
    last_token_idx = actual_seq_length - 1
    rep = latent.pre_acts[last_token_idx, :]

    return rep, actual_seq_length




# %%
name: str = "Circuit Exploration"

# %%
import torch

from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model

from transformers import GPT2Model, GPT2Tokenizer
import transformer_lens

from config import config

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = 'gpt2-small'
model = transformer_lens.HookedTransformer.from_pretrained(model_name)

# Run the model and get logits and activations
# logits, activations = model.run_with_cache("Hello World")

# %%
# print(f"Logits shape: {logits.shape}")

tokens = model.to_tokens("Hello, world!")

model = patchable_model(
    model, factorized=True, slice_output="last_seq", separate_qkv=True, device=device
)

ablations = src_ablations(model, tokens, AblationType.ZERO)


# %%
patch_edges = [
    "Resid Start->MLP 2",
    "MLP 2->A2.4.Q",
    "A2.4->Resid End",
]
with patch_mode(model, ablations, patch_edges):
    patched_out = model(tokens)

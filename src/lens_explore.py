# %%
# https://stackoverflow.com/a/78136410
# Fixes ModuleNotFoundError: No module named 'distutils'
import setuptools.dist
import torch
import transformer_lens
from transformers import GPT2Model, GPT2Tokenizer

from config import config
from visuals import visualize_activations

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model_name = "gpt2-small"
model = transformer_lens.HookedTransformer.from_pretrained(model_name)

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")

# %%
print(f"Logits shape: {logits.shape}")
print(f"Activations: {activations.keys()}")


# %%
visualize_activations(activations)

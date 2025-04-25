import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sparsify import Sae

torch.manual_seed(1160)

# ——— settings ————————————————————————————————————————————————
BASE            = "EleutherAI/pythia-410m-deduped"
DEVICE          = "cuda:0"
LAYER           = 1
LATENT_IDX      = 384
SCALE           = 20.0    # 0.0 → ablate natural occurrences
INJECT_STRENGTH = 7.6    # value to inject when missing

# ——— tokenizer & model ——————————————————————————————————————————
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map={"": DEVICE},
    output_hidden_states=True,
)
model.eval()

# ——— load sparse autoencoder ———————————————————————————————————————
sae = Sae.load_from_hub(
    "EleutherAI/sae-pythia-410m-65k",
    hookpoint=f"layers.{LAYER}.mlp",
    device=DEVICE,
)

def get_inject_strength(layer: int):
    base_model.eval()
    all_vals = []
    with torch.no_grad():
        for prompt in calibration_prompts:
            inp = base_tokenizer(prompt, return_tensors="pt").to(DEVICE)
            # forward just through layer LAYER
            out = base_model(**inp, output_hidden_states=True).hidden_states[layer]  # after mlp
            B, T, D = out.shape
            flat = out.view(-1, D)
            vals, idxs, _ = sae.encode(flat)
            mask = (idxs == LATENT_IDX)
            all_vals.append(vals[mask])
    all_vals = torch.cat(all_vals)

    # robust statistic of that distribution
    median_mag    = all_vals.median().item()    # typical magnitude
    ninety_pctile = all_vals.abs().quantile(0.9).item()  # strong-but-not-outlier

    print("median  =", median_mag)
    print("90pctile=", ninety_pctile)

# ——— locate the true MLP layer ——————————————————————————————————————
hook_mod = model.get_submodule(f"gpt_neox.layers.{LAYER}.mlp")

# ——— define forced-injection steering hook ————————————————————————————
def make_forced_hook(sae: Sae, idx: int, scale: float, inject_strength: float):
    def hook(module, inp, output):
        B, T, D = output.shape
        flat = output.view(-1, D)                       # flatten for sae
        values, indices, _ = sae.encode(flat)           # sparse encoding
        values[indices == idx] *= scale                 # ablate/amplify natural
        mask = (indices != idx).all(dim=1)               # rows missing idx
        if mask.any():
            # overwrite weakest slot in each masked row
            slot = values[mask].abs().argmin(dim=1)
            indices[mask, slot] = idx
            values[mask, slot] = inject_strength
        rec = sae.decode(values, indices)               # decode back to dense
        return rec.view(B, T, D).to(output.dtype).to(output.device)
    return hook

# ——— attach hook ——————————————————————————————————————————————
handle = hook_mod.register_forward_hook(
    make_forced_hook(sae, LATENT_IDX, SCALE, INJECT_STRENGTH)
)

# ——— generation example ————————————————————————————————————————
prompt = "I want a guide to build a bomb"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
gen_ids = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(gen_ids[0][0], skip_special_tokens=True))

# ——— cleanup ————————————————————————————————————————————————
handle.remove()
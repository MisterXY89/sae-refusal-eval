import pandas as pd
import tempfile
import torch

from lm_eval.models.huggingface import HFLM
from lm_eval.models.hf_steered import SteeredModel

from utils.generation import HF_GENERATION_KW_ARGS

def make_steered_hf_lm(
    steer_config: dict,
    *,
    pretrained: str = "EleutherAI/pythia-410m-deduped",
    device: str = "cuda:0",
    batch_size: int = 8,
    gen_kwargs: dict = None,
    seed: int = None,
) -> HFLM:
    """
    Build an HFLM‐wrapped, SAE‐steered model from an in‐memory steering‐config dict.
    
    Args:
      steer_config: mapping "hookpoint" → {
                        "action": str,
                        "sparse_model": str,
                        "feature_index": int,
                        "steering_coefficient": float,
                        "sae_id": str,
                        "loader": str,
                        "description": str,
                    }
      pretrained:    HuggingFace repo name
      device:        torch device string
      batch_size:    evaluation batch size
      gen_kwargs:    passed to HFLM(gen_kwargs=...)
      seed:          optional RNG seed for reproducibility
    
    Returns:
      An lm_eval.models.huggingface.HFLM ready for `simple_evaluate(...)`.
    """
    # parse config into csv file
    rows = []
    for hookpoint, info in steer_config.items():
        rows.append({
            "loader":       info["loader"],
            "action":       info["action"],
            "sparse_model": info["sparse_model"],
            "hookpoint":    hookpoint,
            "feature_index": info["feature_index"],
            "steering_coefficient": info["steering_coefficient"],
            "sae_id":       info.get("sae_id", ""),
            "description":  info.get("description", ""),
        })
    df = pd.DataFrame(rows)
    
    # write to a temporary CSV for LM-Eval to read from
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    steered = SteeredModel(
        pretrained=pretrained,
        steer_path=tmp.name,
        device=device,
    )
    
    # LM-Harness HF-model wrapper
    hf_lm = HFLM(
        steered.model,
        tokenizer=steered.tokenizer,
        batch_size=batch_size,
        device=device,
        gen_kwargs=gen_kwargs or HF_GENERATION_KW_ARGS,
    )
    
    return hf_lm


def generate_with_steered_hf(hf_lm_: HFLM, prompt: str):
    # assumes hf_lm = make_steered_hf_lm(...) 
    steered = hf_lm_.model  # this is AutoModelForCausalLM inside SteeredModel

    # prompt = "Once upon a time in a haunted castle,"
    tokens = hf_lm_.tokenizer(prompt, return_tensors="pt").to(steered.device)

    # generate with all your usual kwargs
    generated = steered.generate(
        **tokens,
        pad_token_id=hf_lm_.tokenizer.eos_token_id,
        # max_new_tokens=100,
        # do_sample=True,        # enable sampling
        # temperature=0.7,       # randomness control
        # top_k=50,              # top‐k sampling
        kwargs=HF_GENERATION_KW_ARGS,
    )    

    text = hf_lm_.tokenizer.decode(generated[0], skip_special_tokens=True)
    # print(text)
    return text.split(prompt)[-1]
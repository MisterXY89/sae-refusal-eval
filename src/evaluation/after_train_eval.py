import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparsify import Sae
from datasets import load_dataset
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from typing import Dict, Optional
import warnings

# --- Configuration ---
CACHE_DIR = "/share/tilman.kerl/huggingface"
os.environ["HF_HOME"] = CACHE_DIR
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _load_model_and_tok(model_name: str, dev: str = DEV):
    """Loads a Hugging Face model and tokenizer with validation."""
    print(f"Loading model '{model_name}' on device '{dev}'...")
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            return_dict_in_generate=True
        ).to(dev).eval()
        
        return tok, lm
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def id_fn(inp):
    """Identity function for dataset text processing."""
    return inp

def stream_dataset(dset, tok, max_length: int, text_field_name: str, text_field_fn):
    """Generator function to yield tokenized batches from a dataset."""
    for ex in dset:
        try:
            yield tok(
                text_field_fn(ex[text_field_name]),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(DEV)
        except Exception as e:
            warnings.warn(f"Skipping batch due to tokenization error: {e}")
            continue


def validate_sae_compatibility(sae, model_dim: int):
    """Validates that SAE dimensions match the model."""
    if hasattr(sae, 'd_in') and sae.d_in != model_dim:
        raise ValueError(f"SAE input dim ({sae.d_in}) doesn't match model dim ({model_dim})")


def collect_activations_and_reconstructions_chunked(
    lm: AutoModelForCausalLM,
    sae: torch.nn.Module,
    tok: AutoTokenizer,
    ds,
    layer_idx: int,
    max_len: int,
    text_field_name: str = "text",
    text_field_fn = id_fn
):
    """
    Memory-efficient version that processes data in chunks and computes metrics incrementally.
    Uses a vectorized Welford's algorithm for numerically stable variance calculation.
    """
    num_latents = sae.num_latents
    latent_fired = torch.zeros(num_latents, dtype=torch.bool, device="cpu")
    model_dim = lm.config.hidden_size 
    
    # --- Welford's algorithm variables for stable variance calculation ---
    agg_mean = torch.zeros(model_dim, device="cpu")
    agg_m2 = torch.zeros(model_dim, device="cpu")
    n_total_tokens = 0
    # ---

    # --- Running statistics for other metrics ---
    sum_mse = 0.0
    sum_cos_sim = 0.0
    nz_count = 0
    nz_total = 0
    n_samples_for_mse = 0 # Use a separate counter for MSE averaging

    print(f"Collecting activations from layer {layer_idx} (chunked processing)...")
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(stream_dataset(ds, tok, max_len, text_field_name, text_field_fn), total=len(ds))):
            out = lm(**batch)
            h = out.hidden_states[layer_idx + 1].flatten(0, 1).to(DEV)
            
            # Skip empty batches which can occur with streaming/tokenization errors
            if h.shape[0] == 0:
                continue

            # --- Vectorized Welford's Algorithm Update ---
            h_cpu = h.cpu()
            n_new_tokens = h_cpu.shape[0]
            n_old_tokens = n_total_tokens
            n_total_tokens += n_new_tokens

            new_mean = h_cpu.mean(dim=0)
            delta = new_mean - agg_mean
            
            # Update aggregate mean
            agg_mean += delta * (n_new_tokens / n_total_tokens)
            
            # Update aggregate M2 (sum of squared differences)
            new_m2 = ((h_cpu - new_mean) ** 2).sum(dim=0)
            agg_m2 += new_m2 + (delta ** 2) * (n_old_tokens * n_new_tokens / n_total_tokens)
            # --- End of Welford's Update ---
            
            # --- SAE forward pass ---
            latent_acts, latent_indices, r = None, None, None
            try:
                if hasattr(sae, 'forward'):
                    output = sae(h)
                    if hasattr(output, 'latent_acts') and hasattr(output, 'sae_out'):
                        latent_acts = output.latent_acts
                        latent_indices = getattr(output, 'latent_indices', None)
                        r = output.sae_out
                    elif isinstance(output, tuple):
                        r, latent_acts = output[:2]
                        if len(output) > 2:
                            latent_indices = output[2]
                    else: # Fallback: assume output is reconstruction
                        r = output
                else: # Fallback to original sparsify interface
                    latent = sae.encode(h)
                    if hasattr(latent, 'top_acts') and hasattr(latent, 'top_indices'):
                        r = sae.decode(latent.top_acts, latent.top_indices)
                        latent_acts = latent.top_acts
                        latent_indices = latent.top_indices
                    else:
                        r = sae.decode(latent)
                        latent_acts = latent
            except Exception as e:
                print(f"Warning: Error in SAE forward pass, skipping batch: {e}")
                continue

            # --- Update running statistics for other metrics ---
            batch_token_count = h.shape[0]
            n_samples_for_mse += batch_token_count
            
            sum_mse += ((h - r) ** 2).mean().item() * batch_token_count
            sum_cos_sim += torch.nn.functional.cosine_similarity(h, r, dim=-1).mean().item() * batch_token_count
            
            # --- Sparsity stats ---
            if latent_acts is not None:
                if latent_indices is not None: # Top-k SAE
                    total_possible = batch_token_count * num_latents
                    actual_active = (latent_acts.abs() > 1e-8).sum().item()
                    nz_count += actual_active
                    nz_total += total_possible
                    
                    unique_indices = torch.unique(latent_indices.flatten()).cpu()
                    valid_indices = unique_indices[unique_indices < num_latents]
                    latent_fired[valid_indices] = True
                else: # Dense SAE
                    nz_count += (latent_acts.abs() > 1e-8).sum().item()
                    nz_total += latent_acts.numel()
                    if latent_acts.shape[-1] == num_latents:
                        fired_features = (latent_acts.abs() > 1e-8).any(dim=0).cpu()
                        latent_fired |= fired_features
            
            if i % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Compute final metrics from running statistics ---
    avg_mse = sum_mse / n_samples_for_mse if n_samples_for_mse > 0 else 0.0
    avg_cos_sim = sum_cos_sim / n_samples_for_mse if n_samples_for_mse > 0 else 0.0

    # Calculate final variance from Welford's aggregates
    if n_total_tokens < 2:
        final_var = 0.0
    else:
        # Variance for each feature dimension, then average across all dimensions
        final_var = (agg_m2 / (n_total_tokens - 1)).mean().item()
    
    return avg_mse, avg_cos_sim, final_var, nz_count, nz_total, latent_fired


def _extract_training_params(sae_checkpoint_dir: str) -> str:
    checkpoint_name = Path(sae_checkpoint_dir).parts[-2] # Gets the 'smollm...-k' part
    parts = checkpoint_name.split('-')
    
    # Initialize params with defaults in case parsing fails
    params = {
        'train_dataset': 'unknown',
        'token_size': 'unknown',
        'expansion_factor': 'unknown',
        'k_value': 'unknown'
    }
    # Safely parse the parts based on the expected structure
    try:
        # Example: smollm2-sparsify-lmsys-419M-token-18-layers-8-expansion-64-k
        params['train_dataset'] = parts[2]
        params['token_size'] = parts[3]
        params['expansion_factor'] = int(parts[7])
        params['k_value'] = int(parts[9])
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse all metadata from checkpoint name '{checkpoint_name}'. Using defaults. Error: {e}")

    return params


def post_train_eval(
    sae_checkpoint_dir: str,
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    eval_dataset: str = "datablations/c4-filter-small",
    split: str = "train",
    layer_idx: int = 18,
    max_len: int = 256,
    max_samples: int = 1000,
    text_field_name: str = "text",
    text_field_fn = id_fn
):
    """
    Main function to run the post-training evaluation of an SAE using chunked processing.
    """
    print("=== Starting SAE Evaluation ===")
    print(f"SAE Checkpoint Dir: {sae_checkpoint_dir}")
    print(f"Base Model: {model_name}")
    print(f"Evaluation Dataset: {eval_dataset} (split: {split})")
    print(f"Layer Index: {layer_idx}, Max Samples: {max_samples}, Max Length: {max_len}\n")

    # "smollm2-sparsify-PRE-419M-token-6_25-layers-16-expansion-64-k/layers.6"

    _fn_lookup = {
        "EQ": "MIX",
        "INS": "MIX",
        "PRE": "MIX",
        "lmsys": "LMSYS", 
    }
    
    train_ds_token = sae_checkpoint_dir.split("-")[2]
    train_ds_folder_name = _fn_lookup[train_ds_token]
    
    
    # Validate checkpoint exists    
    full_checkpoint_path = Path(f"/home/tilman.kerl/mech-interp/src/train/{train_ds_folder_name}/checkpoints/{sae_checkpoint_dir}")
    if not full_checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {full_checkpoint_path}")
    
    # Load models and SAE
    tok, lm = _load_model_and_tok(model_name)
    
    print(f"Loading SAE from '{full_checkpoint_path}'...")
    sae = Sae.load_from_disk(str(full_checkpoint_path)).to(DEV).eval()
    
    # Validate compatibility
    model_dim = lm.config.hidden_size
    validate_sae_compatibility(sae, model_dim)
    
    print("Loading dataset...")
    ds = load_dataset(
        eval_dataset,
        split=f"{split}[:{max_samples}]",
        streaming=False, # Must be False to get total for tqdm
        cache_dir=CACHE_DIR
    )

    # --- Run Memory-efficient chunked processing ---
    avg_mse, avg_cos_sim, final_var, nz_count, nz_total, latent_fired = \
        collect_activations_and_reconstructions_chunked(
            lm=lm, sae=sae, tok=tok, ds=ds, 
            layer_idx=layer_idx, max_len=max_len,
            text_field_name=text_field_name, text_field_fn=text_field_fn
        )
    
    # --- Final Metric Calculation ---
    fvu = avg_mse / max(final_var, 1e-12)
    ev = 1.0 - fvu
    act_sparsity = nz_count / max(nz_total, 1)
    
    with torch.no_grad():
        w_sp = (sae.W_dec.abs() < 1e-6).float().mean().item()
    
    total_latents = len(latent_fired)
    dead_count = int((~latent_fired).sum())
    dead_pct = dead_count / total_latents if total_latents > 0 else 0.0
    
    # --- Compile, Print, and Save Metrics ---

    training_params = _extract_training_params(sae_checkpoint_dir)
    
    metrics = {
        "explained_variance": ev,
        "fraction_var_unexplained": fvu,
        "mse": avg_mse,
        "cosine_similarity": avg_cos_sim,
        "activation_sparsity_l0": act_sparsity,
        "weight_sparsity": w_sp,
        "dead_features_pct": dead_pct,
        "dead_features_count": dead_count,
        "total_latents": total_latents,
                
        "eval_dataset": eval_dataset,
        "model": model_name,
        "layer": layer_idx,
            
        "sae_train_dataset": training_params['train_dataset'],
        "sae_token_size": training_params['token_size'],
        "sae_expansion_factor": training_params['expansion_factor'],
        "sae_k_value": training_params['k_value'],
        "sae_checkpoint_name": sae_checkpoint_dir,
    }

    print("\n--- SAE Metrics (Chunked Processing) ---")
    print(f"Explained Variance       : {metrics['explained_variance']:.4f}")
    print(f"Fraction of Var Unexpl.  : {metrics['fraction_var_unexplained']:.4f}")
    print(f"Mean Squared Error (MSE) : {metrics['mse']:.2e}")
    print(f"Cosine Similarity        : {metrics['cosine_similarity']:.4f}")
    print(f"Activation Sparsity (L0) : {metrics['activation_sparsity_l0']:.4%}")
    print(f"Weight Sparsity          : {metrics['weight_sparsity']:.4%}")
    print(f"Dead Latent Features     : {metrics['dead_features_pct']:.2%} ({metrics['dead_features_count']}/{metrics['total_latents']})")
    
    # --- Save results to JSON ---
    results_dir = Path("results/saes")
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_ds_token = eval_dataset.split("/")[1]
    
    # Use the name of the checkpoint directory for the json file
    basename = sae_checkpoint_dir.split("/layers.")[0]
    json_filename = f"{basename}_layer_{layer_idx}_ds_{eval_ds_token}.json"
    save_path = results_dir / json_filename
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE post-training metrics using chunked processing.")
    parser.add_argument("--sae_checkpoint", required=True, help="Path to SAE checkpoint directory (the name of the folder).")
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM-135M", help="Base model name.")
    parser.add_argument("--dataset", default="datablations/c4-filter-small", help="Evaluation dataset.")
    parser.add_argument("--layer_idx", type=int, default=8, help="Layer index to evaluate.")
    parser.add_argument("--max_samples", type=int, default=10000, help="Maximum number of samples for evaluation.")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length for tokenization.")
    
    args = parser.parse_args()
    
    post_train_eval(
        sae_checkpoint_dir=args.sae_checkpoint,
        model_name=args.model_name,
        eval_dataset=args.dataset,
        layer_idx=args.layer_idx,
        max_samples=args.max_samples,
        max_len=args.max_len,
    )


if __name__ == '__main__':
    main()

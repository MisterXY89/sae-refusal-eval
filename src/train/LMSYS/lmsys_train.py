from sae_lens import (
    SAETrainingRunner, LanguageModelSAERunnerConfig,
    upload_saes_to_huggingface
)

from datasets import load_dataset

# ——— Toggle TEST_MODE ——————————————————————————————————————————————————
# If True → tiny run, no logging, no uploads. If False → full run + HuggingFace upload.
TEST_MODE = False
# MisterXY89/lmsys-chat-1m-english-tokenized

def run():
    # —— Schedule & batch sizing —————————————————————————————————————
    if TEST_MODE:
        total_steps = 10
        batch_size_tokens = 512
        dataset_path = "karpathy/tiny_shakespeare"
        log_to_wandb = False
        wandb_project = None
        n_checkpoints = 0
        checkpoint_path = None

        # disable all precision tricks to avoid dtype mismatches
        use_autocast = False
        use_autocast_lm = False
        use_compile_sae = False
        use_compile_llm = False
        sae_dtype = "float32"
    else:
        total_steps = 75_000
        batch_size_tokens = 3072 # 2048 #4_096 
        # dataset_path = "wikitext/wikitext-103-raw-v1"
        # dataset_path = "apollo-research/monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b"    	
        dataset_path = "MisterXY89/lmsys-chat-1m-english-tokenized"
        log_to_wandb = True
        wandb_project = "MA-sae-train"
        n_checkpoints = 5
        checkpoint_path = "./checkpoints"
        
        use_autocast = False #True
        use_autocast_lm = False
        use_compile_sae = False
        use_compile_llm = False
        sae_dtype = "float32"  # or "bfloat16" if you want to match the model

    total_training_tokens = total_steps * batch_size_tokens
    lr_warmup_steps = int(0.05 * total_steps)
    lr_decay_steps = total_steps
    l1_warmup_steps = int(0.40 * total_steps)

    cfg = LanguageModelSAERunnerConfig(
        # — data & model hooks —
        model_name = "EleutherAI/pythia-410m-deduped",
        hook_name = "blocks.10.hook_resid_post",
        hook_layer = 10,
        d_in = 1024,
        dataset_path = dataset_path,
        is_dataset_tokenized = True, #False,
        streaming = True,

        # — SAE architecture & sparsity —
        architecture = "standard",
        expansion_factor = 8, # 16,
        l1_coefficient = 0.5, #2.0,
        l1_warm_up_steps = l1_warmup_steps,
        normalize_activations = "expected_average_only_in",
        mse_loss_normalization = "none", # "layer",

        # — init & symmetry —
        b_dec_init_method  = "zeros",
        init_encoder_as_decoder_transpose = True,
        decoder_heuristic_init = False,

        # — optimization & scheduling —
        lr = 1e-4, # 5e-5,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        lr_scheduler_name = "cosineannealing",
        lr_warm_up_steps = lr_warmup_steps,
        lr_decay_steps = lr_decay_steps,

        # — context & batch sizing —
        context_size = 128, # pre-token with 128 (not 512) -> maybe need to adopt,
        train_batch_size_tokens = batch_size_tokens,

        # — logging, checkpoints & precision —
        training_tokens = total_training_tokens,
        feature_sampling_window = 5000, # 1_000,
        log_to_wandb = log_to_wandb,
        wandb_project = wandb_project,
        wandb_log_frequency = 100,
        n_checkpoints = n_checkpoints,
        checkpoint_path = checkpoint_path,
        compile_sae = use_compile_sae,
        compile_llm = use_compile_llm,
        autocast = use_autocast,
        autocast_lm = use_autocast_lm,
        device = "cuda:0",
        seed = 42,
        dtype = sae_dtype,
    )

    print(f"{'TEST_MODE' if TEST_MODE else 'FULL_RUN'} ➞ steps={total_steps}, dataset={dataset_path}")

    
    # — run training —
    train_ds = load_dataset(
        "MisterXY89/lmsys-chat-1m-english-tokenized",
        split="train",
        streaming=True,            # keep streaming mode if desired
        trust_remote_code=True,    # if your dataset requires it
    )
    sparse_autoencoder = SAETrainingRunner(cfg, override_dataset=train_ds).run()
    # sparse_autoencoder = SAETrainingRunner(cfg).run()

    if not TEST_MODE:
        # — Upload to HuggingFace —
        HF_USER    = "MisterXY89"
        MODEL_NAME = "sae-pythia-410m-deduped-v0"
        sae_dict   = { cfg.hook_name: sparse_autoencoder }
        repo_id    = f"{HF_USER}/{MODEL_NAME}"

        upload_saes_to_huggingface(
            sae_dict,
            hf_repo_id=repo_id,
#            private=True
        )
        print(f"✅ Uploaded SAE to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    run()

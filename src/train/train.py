
from sae_lens import (
    SAETrainingRunner, LanguageModelSAERunnerConfig, 
    upload_saes_to_huggingface
)

def run():
    # —— Schedule & batch sizing —————————————————————————————————————
    total_steps            = 75_000                                    # ∼75 K gradient updates
    batch_size_tokens      = 4_096                                     # 4 K tokens per step to keep an A100 busy
    total_training_tokens  = total_steps * batch_size_tokens          # ≈300 M tokens


    # —— Learning‐rate & sparsity warmups —————————————————————————————
    lr_warmup_steps  = int(0.1 * total_steps)                          # 10% warmup avoids dead features
    lr_decay_steps   = int(0.2 * total_steps)                          # 20% decay to stabilize later
    l1_warmup_steps  = int(0.05 * total_steps)                         # 5% warmup on the L1 penalty


    # —— SAE‐Lens configuration ———————————————————————————————————————
    cfg = LanguageModelSAERunnerConfig(
        # — Data & model hooks —
        model_name                    = "EleutherAI/pythia-410m-deduped",
        hook_name                     = "blocks.4.hook_mlp_out",  # intervene on layer 4 MLP output
        hook_layer                    = 4,
        d_in                          = 1024,                    # MLP hidden‐size for Pythia-410m
        # dataset_path                  = "EleutherAI/pile",       # large, diverse language corpus
        dataset_path                  = "karpathy/tiny_shakespeare", 
        is_dataset_tokenized          = False,
        streaming                     = True,                     # stream directly from HF

        # — SAE architecture & sparsity —
        architecture                  = "standard",               # L1‐penalized SAE
        expansion_factor              = 16,                       # 16× overcomplete basis (≈32 K latents)
        l1_coefficient                = 5.0,                      # medium‐strength sparsity
        l1_warm_up_steps              = l1_warmup_steps,
        mse_loss_normalization        = None,                     # raw MSE

        # — Decoder & encoder initialization —
        b_dec_init_method             = "zeros",                  # start decoder at zero
        init_encoder_as_decoder_transpose = True,                # symmetry speeds convergence
        decoder_heuristic_init        = False,                     # geometric‐median bias helps stability

        # — Optimization & scheduling —
        lr                            = 5e-5,                     # standard Adam LR
        adam_beta1                    = 0.9,
        adam_beta2                    = 0.999,
        lr_scheduler_name             = "constant",
        lr_warm_up_steps              = lr_warmup_steps,
        lr_decay_steps                = lr_decay_steps,

        # — Context & batch sizing —
        train_batch_size_tokens       = batch_size_tokens,
        context_size                  = 256,                      # 256-token windows

        # — Reporting & logging —
        training_tokens               = total_training_tokens,
        feature_sampling_window       = 1_000,                    # report every 1 K steps
        log_to_wandb                  = True,
        wandb_project                 = "MA-sae-train",
        wandb_log_frequency           = 50,

        # — Miscellaneous —
        device                        = "cuda:0",
        seed                          = 42,
        n_checkpoints                 = 0,
        dtype                         = "float32"
    )

    # — Run!
    sparse_autoencoder = SAETrainingRunner(cfg).run()

    # - Upload to HF
    sae_dict = {
        cfg.hook_name: sparse_autoencoder
    }

    # WRITE TOKEN: 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
    upload_saes_to_huggingface(
        saes=sae_dict,
        hf_repo_id="MisterXY89/sae-pythia-410m-deduped-v1",  # your target repo
        private=True                                         # set True if you want a private repo
    )

    print(f"✅ Uploaded SAE to https://huggingface.co/MisterXY89/sae-pythia-410m-deduped-v1")


if __name__ == "__main__":
    run()

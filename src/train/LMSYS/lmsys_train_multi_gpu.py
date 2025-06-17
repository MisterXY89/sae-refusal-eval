import os
import torch
import torch.distributed as dist

from datasets import load_dataset
from sae_lens import (
    SAETrainingRunner,
    LanguageModelSAERunnerConfig,
    upload_saes_to_huggingface,
    # LoggingConfig
)
from sae_lens.config import LoggingConfig

from sae_lens.saes import TopKTrainingSAEConfig

def main():
    # — init distributed —
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # — config parameters —
    # projected steps for convergence (gao et al.)
    TOTAL_STEPS           = 65_000 # try different strategies
    GLOBAL_BATCH_TOKENS   = 65_536
    WG_SIZE              = dist.get_world_size()
    PER_GPU_TOKENS       = GLOBAL_BATCH_TOKENS // WG_SIZE
    TRAIN_TOKENS         = TOTAL_STEPS * GLOBAL_BATCH_TOKENS
    LR_WARMUP            = int(0.05 * TOTAL_STEPS)

    sae_config = TopKTrainingSAEConfig(
        k                 = 1_024,      # active latents        
        d_sae               = 512 * 8, # # latent dimension (expansion_factor × d_in)
        d_in = 512,
        # expansion_factor  = 8,
    )  # no l1_

    cfg = LanguageModelSAERunnerConfig(
        # model & hook
        model_name = "HuggingFaceTB/SmolLM2-135M",
        hook_name = "blocks.7.hook_resid_post",
        hook_layer = 7,
                        
        # init_encoder_as_decoder_transpose = True,
        # aux_k_loss        = True,
        # mean_center       = True,
        # unit_norm         = True,
        context_size      = 64,

        sae = sae_config,

        # data
        dataset_path = "MisterXY89/lmsys-chat-1m-english-tokenized-smollm-135M",
        is_dataset_tokenized = True,
        streaming = True,

        lr                    = 1e-4,
        adam_beta1            = 0.9,
        adam_beta2            = 0.999,
        lr_scheduler_name     = "cosineannealing",
        lr_warm_up_steps      = LR_WARMUP,
        lr_decay_steps        = TOTAL_STEPS,

        train_batch_size_tokens = PER_GPU_TOKENS,
        training_tokens       = TRAIN_TOKENS,

        logger = LoggingConfig(
            log_to_wandb          = True,
            wandb_project         = "SAE-SmolLM-135M",
            wandb_log_frequency   = 100,
        ),
        n_checkpoints         = 5,
        checkpoint_path       = "./checkpoints",

        compile_sae           = False,
        compile_llm           = False,
        autocast              = True,
        autocast_lm           = True,
        dtype                 = "float32",

        device                = f"cuda:{local_rank}",
        seed                  = 42,

        # use_sparse_kernels    = True,
        # ema_decay             = 0.999,
    )

    if local_rank == 0:
        print(f"→ Starting SAE training: {cfg.model_name}, layer {cfg.hook_layer}")
        print(f"   total_steps={TOTAL_STEPS}, global_batch={GLOBAL_BATCH_TOKENS} tokens, "
              f"per-GPU={PER_GPU_TOKENS} tokens")

    # — data loader —
    train_ds = load_dataset(
        cfg.dataset_path,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )


    sae_model = SAETrainingRunner(cfg, override_dataset=train_ds).run()

    # — only rank 0 uploads —
    if local_rank == 0:
        repo_id = "MisterXY89/sae-smollm-135m-v0"
        upload_saes_to_huggingface(
            {cfg.hook_name: sae_model},
            hf_repo_id=repo_id
        )
        print(f"✅ Uploaded SAE to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
import os
import torch
import torch.distributed as dist

from datasets import load_dataset
from sae_lens import (
    SAETrainingRunner,
    LanguageModelSAERunnerConfig,
    upload_saes_to_huggingface,
    # LoggingConfig
)
from sae_lens.config import LoggingConfig

from sae_lens.saes import TopKTrainingSAEConfig

def main():
    # — init distributed —
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # — config parameters —
    # projected steps for convergence (gao et al.)
    TOTAL_STEPS           = 65_000 # try different strategies
    GLOBAL_BATCH_TOKENS   = 65_536
    WG_SIZE              = dist.get_world_size()
    PER_GPU_TOKENS       = GLOBAL_BATCH_TOKENS // WG_SIZE
    TRAIN_TOKENS         = TOTAL_STEPS * GLOBAL_BATCH_TOKENS
    LR_WARMUP            = int(0.05 * TOTAL_STEPS)

    sae_config = TopKTrainingSAEConfig(
        k                 = 1_024,      # active latents        
        d_sae               = 512 * 8, # # latent dimension (expansion_factor × d_in)
        d_in = 512,
        # expansion_factor  = 8,
    )  # no l1_

    cfg = LanguageModelSAERunnerConfig(
        # model & hook
        model_name = "HuggingFaceTB/SmolLM2-135M",
        hook_name = "blocks.7.hook_resid_post",
        hook_layer = 7,
                        
        # init_encoder_as_decoder_transpose = True,
        # aux_k_loss        = True,
        # mean_center       = True,
        # unit_norm         = True,
        context_size      = 64,

        sae = sae_config,

        # data
        dataset_path = "MisterXY89/lmsys-chat-1m-english-tokenized-smollm-135M",
        is_dataset_tokenized = True,
        streaming = True,

        lr                    = 1e-4,
        adam_beta1            = 0.9,
        adam_beta2            = 0.999,
        lr_scheduler_name     = "cosineannealing",
        lr_warm_up_steps      = LR_WARMUP,
        lr_decay_steps        = TOTAL_STEPS,

        train_batch_size_tokens = PER_GPU_TOKENS,
        training_tokens       = TRAIN_TOKENS,

        logger = LoggingConfig(
            log_to_wandb          = True,
            wandb_project         = "SAE-SmolLM-135M",
            wandb_log_frequency   = 100,
        ),
        n_checkpoints         = 5,
        checkpoint_path       = "./checkpoints",

        compile_sae           = False,
        compile_llm           = False,
        autocast              = True,
        autocast_lm           = True,
        dtype                 = "float32",

        device                = f"cuda:{local_rank}",
        seed                  = 42,

        # use_sparse_kernels    = True,
        # ema_decay             = 0.999,
    )

    if local_rank == 0:
        print(f"→ Starting SAE training: {cfg.model_name}, layer {cfg.hook_layer}")
        print(f"   total_steps={TOTAL_STEPS}, global_batch={GLOBAL_BATCH_TOKENS} tokens, "
              f"per-GPU={PER_GPU_TOKENS} tokens")

    # — data loader —
    train_ds = load_dataset(
        cfg.dataset_path,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )


    sae_model = SAETrainingRunner(cfg, override_dataset=train_ds).run()

    # — only rank 0 uploads —
    if local_rank == 0:
        repo_id = "MisterXY89/sae-smollm-135m-v0"
        upload_saes_to_huggingface(
            {cfg.hook_name: sae_model},
            hf_repo_id=repo_id
        )
        print(f"✅ Uploaded SAE to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()

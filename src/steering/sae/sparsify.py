import torch

def steer_sae_features(
    sae,
    feature_indices,
    factor=0.0,
    mode="ablate"
):
    """
    Register a one‐time hook on `sae` to steer given latent features.

    Args:
      sae:          an instance of sparsify.Sae
      feature_indices: list[int]  latent feature indices to modify
      factor:       float        scaling factor (only used in 'scale' mode)
      mode:         str          'ablate' to zero out, 'scale' to multiply by factor

    Returns:
      A handle you can call .remove() on to disable steering.
    """

    def _hook(acts, hook):
        # acts: tensor of shape [..., d_sae]
        if mode == "ablate":
            acts = acts.clone()
            acts[..., feature_indices] = 0.0
        elif mode == "scale":
            acts = acts.clone()
            acts[..., feature_indices] *= factor
        else:
            raise ValueError("mode must be 'ablate' or 'scale'")
        return acts

    # register the hook; it runs after SAE’s activation pre‐decode
    handle = sae.hook_sae_acts_post.add_hook(_hook)
    return handle

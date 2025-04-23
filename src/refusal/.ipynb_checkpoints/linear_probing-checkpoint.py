import gc
import os
from typing import Any, Callable, List, Optional

import torch
from transformer_lens import utils as tl_utils


# Following Arditi et al. (2023), we define the refusal direction as the difference in
# model activations for two sets of instructions: one set that is harmful and one set
# that is harmless. The refusal direction is computed as the mean activation of the
# harmful instructions minus the mean activation of the harmless instructions.
def extract_refusal_direction(
    model: Any,
    harmful_inst_train: List[str],
    harmless_inst_train: List[str],
    n_inst_train: int,
    layer: int,
    pos: int,
    model_name: str,
    pythia_template: str,
    tokenize_instructions_fn: Callable[..., torch.Tensor],
    return_direction: bool = True,
    force: bool = False,
) -> Optional[torch.Tensor]:
    """
    computes and optionally returns the "refusal direction" based on differences
    in model activations for two sets of instructions (harmful vs. harmless).

    args:
      model: model with a tokenizer attribute and run_with_cache(...) method
      train: list of instruction strings
      n_inst_train: number of instructions from train to use
      layer: layer index to extract from
      pos: token position in the residual stream
      model_name: string appended to the output file name
      pythia_template: template string passed to the tokenization function
      tokenize_instructions: function(tokenizer, template, instructions) -> torch.Tensor
      return_direction: whether to return the computed direction
      force: whether to recompute even if the file already exists

    returns:
      the refusal direction as a torch.Tensor if return_direction is true, else none.
    """
    save_file = f"refusal_dir_{model_name}_layer_{layer}.pt"
    dir_name = os.path.dirname(save_file)
    if dir_name:  # dir_name is empty if save_file has no directory part
        os.makedirs(dir_name, exist_ok=True)
    if os.path.exists(save_file) and not force:
        print(f"{save_file} exists, skipping computation.")
        if return_direction:
            return torch.load(save_file)
        return None

    # tokenize_instructions_fn = lambda instructions: tokenize_instructions(
    #    tokenizer=model.tokenizer,
    #    instructions=instructions,
    #    template=pythia_template
    # )

    print(f"using {n_inst_train} pairs to compute refusal direction")
    harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:n_inst_train])
    harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:n_inst_train])

    harmful_logits, harmful_cache = model.run_with_cache(
        harmful_toks, names_filter=tl_utils.get_act_name("resid_pre", layer)
    )
    harmless_logits, harmless_cache = model.run_with_cache(
        harmless_toks, names_filter=tl_utils.get_act_name("resid_pre", layer)
    )

    harmful_mean_act = harmful_cache["resid_pre", layer][:, pos, :].mean(dim=0)
    harmless_mean_act = harmless_cache["resid_pre", layer][:, pos, :].mean(dim=0)

    refusal_dir = harmful_mean_act - harmless_mean_act
    refusal_dir = refusal_dir / refusal_dir.norm()

    del harmful_cache, harmless_cache, harmful_logits, harmless_logits
    gc.collect()
    torch.cuda.empty_cache()

    torch.save(refusal_dir, save_file)
    print(f"saved refusal direction to {save_file}")

    if return_direction:
        return refusal_dir
    return None

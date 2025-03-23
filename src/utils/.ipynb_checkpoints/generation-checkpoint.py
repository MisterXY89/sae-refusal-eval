import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, cast

import gc
import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer

# define a type variable for d_act so mypy knows what it refers to
D_Act = TypeVar("D_Act")


def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks: Optional[List[Tuple[str, Callable[..., Any]]]] = None,
) -> List[str]:
    if fwd_hooks is None:
        fwd_hooks = []

    all_toks = torch.zeros(
        (toks.shape[0], toks.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=toks.device,
    )
    all_toks[:, : toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_toks[:, -max_tokens_generated + i] = next_tokens

    # batch_decode returns a list of strings, so cast to List[str]
    decoded = model.tokenizer.batch_decode(
        all_toks[:, toks.shape[1] :], skip_special_tokens=True
    )
    return cast(List[str], decoded)


def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    # callable that takes a list of strings and returns an int tensor
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, "batch_size seq_len"]],
    fwd_hooks: Optional[List[Tuple[str, Callable[..., Any]]]] = None,
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    """
    Generate completions in batches using an optional set of forward hooks.
    """
    if fwd_hooks is None:
        fwd_hooks = []

    generations: List[str] = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i : i + batch_size]
        toks = tokenize_instructions_fn(batch_instructions)
        if not isinstance(toks, torch.Tensor):
            # if a tokenizer returns a special structure, extract .input_ids
            toks = toks.input_ids  # type: ignore

        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
        
        del toks, generation
        torch.cuda.empty_cache()
        gc.collect()

    return generations


def format_instruction(
    instruction: str,
    output: Optional[str] = None,
    include_trailing_whitespace: bool = True,
    template: Optional[str] = None,
) -> str:
    """
    Format a single instruction with an optional output, trailing whitespace, and template.
    """
    if template is None:
        # fallback if no template is given
        template = "{instruction}"

    formatted_instruction = template.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: Optional[List[str]] = None,
    include_trailing_whitespace: bool = True,
    template: Optional[str] = None,
) -> Int[Tensor, "batch_size seq_len"]:
    """
    Convert instructions (and optionally outputs) into token IDs using a specified template.
    """
    if outputs is not None:
        prompts = [
            format_instruction(
                instruction=instr,
                output=out,
                include_trailing_whitespace=include_trailing_whitespace,
                template=template,
            )
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction(
                instruction=instr,
                include_trailing_whitespace=include_trailing_whitespace,
                template=template,
            )
            for instr in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    return result.input_ids  # shape: [batch_size, seq_len]


def act_add_hook(
    activation: Float[Tensor, "... D_Act"],
    hook: HookPoint,
    direction: Float[Tensor, "D_Act"],
    steering_coef: float,
) -> Float[Tensor, "... D_Act"]:
    """
    Add (direction * steering_coef) to the activation in-place.
    """
    activation += steering_coef * direction
    return activation


def direction_ablation_hook(
    activation: Float[Tensor, "... D_Act"], hook: HookPoint, direction: Float[Tensor, "D_Act"]
) -> Float[Tensor, "... D_Act"]:
    """
    Project out 'direction' from the activation, removing its component along that direction.
    """
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... D_Act, D_Act single -> ... single"
        )
        * direction
    )
    return activation - proj


def get_refusal_direction_hooks(
    model: HookedTransformer,
    refusal_dir: torch.Tensor,
    act_add_hook: Callable[..., Any],
    direction_ablation_hook: Callable[..., Any],
    intervention_type: str = "ablation",  # or "addition"
    steering_coef: float = 1.0,
    layer: int = 23,
) -> List[Tuple[str, Callable[..., Any]]]:
    """
    Sets up forward hooks to apply a refusal direction to a model's activations,
    either by 'addition' at a single layer or 'ablation' across all layers.
    Returns a list of (activation_name, hook_fn) pairs for hooking.
    """
    torch.set_grad_enabled(False)

    if intervention_type == "addition":
        # Only apply addition at one layer
        intervention_layers = [layer]
        hook_fn = functools.partial(
            act_add_hook, direction=refusal_dir, steering_coef=steering_coef
        )
    else:
        # 'ablation': apply to all layers
        intervention_layers = list(range(model.cfg.n_layers))
        hook_fn = functools.partial(direction_ablation_hook, direction=refusal_dir)

    fwd_hooks: List[Tuple[str, Callable[..., Any]]] = [
        (tl_utils.get_act_name(act_name, lyr), hook_fn)
        for lyr in intervention_layers
        for act_name in ["resid_pre", "resid_post"]
    ]
    return fwd_hooks

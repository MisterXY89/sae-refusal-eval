from typing import List, Optional

from jaxtyping import Int
from torch import Tensor
from transformers import AutoTokenizer

HF_GENERATION_KW_ARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "max_new_tokens": 24,
}


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

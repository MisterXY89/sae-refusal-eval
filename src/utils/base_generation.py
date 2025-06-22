import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseModelGenerator:
    def __init__(self, checkpoint: str, dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="cuda:0",
            torch_dtype=dtype
        ).eval()

    def generate(self, prompt: str, max_new_tokens: int = 64, **gen_kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs
            )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        clean_response = response.split(prompt)[-1]
        return clean_response

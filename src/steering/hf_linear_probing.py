import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from functools import partial
import os, gc


class RefusalDirectionSteering:
    def __init__(self, model_name, prompt_template=None, dtype=torch.bfloat16):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.prompt_template = prompt_template
        self.direction, self.layer = None, None

    def _format(self, x):
        return self.prompt_template.format(instruction=x.strip()) if self.prompt_template else x

    def _get_token_pos(self, inputs):
        input_ids = inputs['input_ids']
        pad_id = self.tokenizer.pad_token_id
        return (input_ids != pad_id).sum(dim=1) - 1

    def _mean_activations(self, prompts, layer, pos=-1):
        acts = []

        def hook_fn(_, __, out):
            acts.append(out.detach().cpu())

        handle = self.model.model.layers[layer].mlp.register_forward_hook(hook_fn)
        try:
            for p in tqdm(prompts, desc="activations"):
                prompt = self._format(p)
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    _ = self.model(**inputs)
                token_pos = self._get_token_pos(inputs)
                cached = acts.pop()
                idx = token_pos.cpu()
                acts.append(cached[range(len(idx)), idx])
        finally:
            handle.remove()
        return torch.cat(acts).mean(0)

    def compute_direction(self, harmful, harmless, layer, save_path=None, force=False):
        if save_path and os.path.exists(save_path) and not force:
            data = torch.load(save_path)
            self.direction = data['vector']
            self.layer = data['layer']
            return
        h = self._mean_activations(harmful, layer)
        hl = self._mean_activations(harmless, layer)
        self.direction = F.normalize(h - hl, dim=0)
        self.layer = layer
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'vector': self.direction, 'layer': layer}, save_path)

    def _add_hook(self, _, __, out, coef):
        out = out.clone()
        vec = self.direction.to(out.device)
        out[:, -1, :] += coef * vec
        return out

    def _ablate_hook(self, _, __, out):
        out = out.clone()
        vec = F.normalize(self.direction.to(out.device), dim=0)
        proj = torch.einsum("bld,d->bl", out, vec).unsqueeze(-1) * vec
        return out - proj

    def generate(self, prompt, intervention="addition", coef=1.0, max_new_tokens=64, **kwargs):
        if self.direction is None:
            raise ValueError("Must compute or load direction first.")
        if intervention not in {"addition", "ablation"}:
            raise ValueError("intervention must be 'addition' or 'ablation'.")

        hook_fn = partial(self._add_hook, coef=coef) if intervention == "addition" else self._ablate_hook
        handle = self.model.model.layers[self.layer].mlp.register_forward_hook(hook_fn)

        try:
            prompt_text = self._format(prompt)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            input_len = inputs['input_ids'].shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.3,
                    top_k=50, 
                    top_p=0.95,
                    **kwargs
                )
            return self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        finally:
            handle.remove()

    def generate_with_addition_loop(self, prompt, coef=1.0, max_new_tokens=64):
        if self.direction is None:
            raise ValueError("Must compute or load direction first.")

        vec = self.direction.to(self.device)
        prompt_text = self._format(prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        all_ids = input_ids.clone()

        def hook_fn(_, __, out):
            out = out.clone()
            out[:, -1, :] += coef * vec
            return out

        for _ in range(max_new_tokens):
            handle = self.model.model.layers[self.layer].mlp.register_forward_hook(hook_fn)
            with torch.no_grad():
                outputs = self.model(input_ids=all_ids, attention_mask=attention_mask)
            handle.remove()

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        return self.tokenizer.decode(all_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

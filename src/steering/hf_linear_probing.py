import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from functools import partial
import gc

class ActivationSteering:
    """
    A class to handle activation steering for causal language models.
    
    This class provides methods to:
    1. Create a steering vector from pairs of harmless and harmful prompts.
    2. Apply this vector during generation to steer the model's output.
    3. Save and load steering vectors for reuse.
    """
    def __init__(self, model_name_or_path, torch_dtype=torch.bfloat16):
        """
        Initializes the model, tokenizer, and sets the device.
        
        Args:
            model_name_or_path (str): The name or path of the Hugging Face model.
            torch_dtype (torch.dtype, optional): The data type for model weights. 
                                                 Defaults to torch.bfloat16 for performance.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch_dtype
        ).to(self.device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.steering_vector = None
        self.layer_idx = None

    def _get_mean_activations(self, prompts, layer_idx):
        """
        Internal method to calculate the mean activations for a list of prompts at a specific layer.
        """
        activations = []
        
        # Define the hook function inside this method
        def capture_hook_fn(module, inp, out):
            # The output of a DecoderLayer is a tuple, with hidden_states as the first element
            activations.append(out[0][:, -1, :].detach().cpu())

        # Register the hook
        hook_handle = self.model.model.layers[layer_idx].register_forward_hook(capture_hook_fn)
        
        try:
            for prompt in tqdm(prompts, desc="Collecting activations"):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model(**inputs)
        finally:
            hook_handle.remove() # Ensure the hook is always removed

        return torch.cat(activations).mean(dim=0)

    def create_steering_vector(self, harmless_prompts, harmful_prompts, layer_idx):
        """
        Creates a steering vector by contrasting activations from harmless and harmful prompts.
        
        Args:
            harmless_prompts (list[str]): A list of prompts that exemplify the desired behavior.
            harmful_prompts (list[str]): A list of prompts that exemplify the undesired behavior.
            layer_idx (int): The index of the decoder layer to extract activations from.
        """
        self.layer_idx = layer_idx
        
        print(f"Creating steering vector at layer {layer_idx}...")
        
        pos_activations = self._get_mean_activations(harmless_prompts, layer_idx)
        neg_activations = self._get_mean_activations(harmful_prompts, layer_idx)
        
        self.steering_vector = F.normalize(pos_activations - neg_activations, dim=0)
        
        print("Steering vector created and normalized.")
        gc.collect() # Clean up memory
        torch.cuda.empty_cache()

    def save_vector(self, filepath):
        """Saves the steering vector and its layer index to a file."""
        if self.steering_vector is None:
            raise ValueError("No steering vector to save. Please create one first.")
        torch.save({
            'vector': self.steering_vector,
            'layer_idx': self.layer_idx
        }, f"refusal_dirs/smolLM/{filepath}")
        print(f"Steering vector saved to {filepath}")

    def load_vector(self, filepath):
        """Loads a steering vector and its layer index from a file."""
        data = torch.load(filepath)
        self.steering_vector = data['vector']
        self.layer_idx = data['layer_idx']
        print(f"Steering vector loaded from {filepath} (for layer {self.layer_idx})")

    def _steering_hook_fn(self, module, inp, out, steering_coef):
        """
        Internal hook function to apply the steering vector during generation.
        """
        # The output of the MLP block is a single tensor
        hidden_states = out
        hs_clone = hidden_states.clone()
        
        if self.steering_vector is None:
            return hs_clone

        vec = self.steering_vector.to(hs_clone.device)
        
        if hs_clone.ndim == 3: # Prefill stage
            hs_clone[:, -1, :] += steering_coef * vec
        elif hs_clone.ndim == 2: # Decode stage
            hs_clone += steering_coef * vec
            
        return hs_clone

    def generate(self, prompt, steering_coef=1.5, **kwargs):
        """
        Generates text from a prompt, applying the steering vector.
        
        Args:
            prompt (str): The input prompt for the model.
            steering_coef (float, optional): The coefficient to scale the steering vector.
                                             A value of 0.0 disables steering. Defaults to 1.5.
            **kwargs: Additional arguments for the model's generate method 
                      (e.g., max_new_tokens, do_sample, temperature).
                      
        Returns:
            str: The generated text.
        """
        if self.steering_vector is None and steering_coef != 0:
            raise ValueError("No steering vector loaded or created. Cannot generate with steering.")
        
        # Set default generation kwargs if not provided
        kwargs.setdefault('max_new_tokens', 100)
        kwargs.setdefault('pad_token_id', self.tokenizer.eos_token_id)
        kwargs.setdefault('do_sample', False)

        hook_handle = None
        if steering_coef != 0:
            # Use functools.partial to pass the coefficient to the hook
            steer_with_coef = partial(self._steering_hook_fn, steering_coef=steering_coef)
            hook_handle = self.model.model.layers[self.layer_idx].mlp.register_forward_hook(steer_with_coef)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        finally:
            if hook_handle is not None:
                hook_handle.remove() # Always ensure the hook is removed
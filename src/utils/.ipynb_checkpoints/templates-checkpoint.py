TEMPLATE = """You are a helpful chatbot.<|endoftext|>{instruction}<|endoftext|>"""

# PYTHIA_TEMPLATE = """USER: {instruction}
# ASSISTANT:"""

PYTHIA_TEMPLATE = """Question: {instruction}
Answer:"""


LLAMA_BASE_TEMPLATE = """USER: {instruction}
ASSISTANT:"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant"""

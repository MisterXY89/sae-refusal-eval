import io
from typing import List, Tuple

import pandas as pd
import requests  # type: ignore
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def get_harmful_instructions() -> Tuple[List[str], List[str]]:
    url: str = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    )
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions: List[str] = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions() -> Tuple[List[str], List[str]]:
    hf_path: str = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path)
    instructions: List[str] = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

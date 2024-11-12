import os
from dataclasses import dataclass, field
from typing import Dict

import pretty_errors
from dotenv import load_dotenv

pretty_errors.activate()
load_dotenv()

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
SEED = 1160

@dataclass
class Paths:
    root: str = ROOT_DIR
    data: str = DATA_DIR
    src: str = SRC_DIR

@dataclass
class Credentials:
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    openai_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # openai ...

@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    credentials: Credentials = field(default_factory=Credentials)
    seed: int = SEED

class ConfigSingleton:
    _instance = None

    @classmethod
    def get_instance(cls) -> Config:
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

# global instance --> avoid repeated instantiation
config = ConfigSingleton.get_instance()


if __name__ == "__main__":
    print(f"Root directory: {config.paths.root}")
    print(f"Data directory: {config.paths.data}")
    print(f"Seed: {config.seed}")

    # check if the singleton is working
    config2 = ConfigSingleton.get_instance()
    assert config is config2

    # example access
    print(f"HF Token: {config.credentials.hf_token}")

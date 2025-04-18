"""
Configuration classes for the simulation package.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for the model used in simulations."""
    model_name: str = "all-MiniLM-L6-v2"

@dataclass
class SimulationConfig:
    """Configuration for running simulations."""
    num_bags: int = 100
    min_bag_size: int = 15
    max_bag_size: int = 200
    show_examples: int = 3
    top_k: int = 10

@dataclass
class PreTrainedModelConfig:
    """Configuration for pre-trained models."""
    models: Dict[str, str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = {
                "all-MiniLM-L6-v2": "Good balance of speed and quality",
                "all-mpnet-base-v2": "Higher quality but slower",
                "multi-qa-mpnet-base-dot-v1": "Optimized for semantic search",
                "paraphrase-multilingual-mpnet-base-v2": "Multilingual support"
            }

# Default configurations
DEFAULT_PRETRAINED_CONFIG = PreTrainedModelConfig()
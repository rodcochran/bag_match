from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ModelConfig:
    """Configuration for the sentence transformer model."""
    model_names: List[str] = None
    cache_folder: str = "./models"
    device: str = "mps"  # or "cuda" or "cpu"
    embedding_cache_size: int = 1000
    text_cache_size: int = 1024

    def __post_init__(self):
        if self.model_names is None:
            self.model_names = ["all-MiniLM-L6-v2"]

@dataclass
class SimulationConfig:
    """Configuration for the simulation process."""
    num_bags: int = 100
    min_bag_size: int = 15
    max_bag_size: int = 200
    show_examples: int = 3
    top_k: int = 10
    similarity_measures: List[str] = None

    def __post_init__(self):
        if self.similarity_measures is None:
            self.similarity_measures = [
                "jaccard",
                "average_pairwise",
                "concatenated",
                "pre_trained"
            ]

@dataclass
class PreTrainedModelConfig:
    """Configuration for pre-trained models."""
    models: Dict[str, str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = {
                "all-mpnet-base-v2": "Higher quality but slower",
                "multi-qa-mpnet-base-dot-v1": "Optimized for semantic search",
                "paraphrase-multilingual-mpnet-base-v2": "Multilingual support"
            }

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_PRETRAINED_CONFIG = PreTrainedModelConfig() 
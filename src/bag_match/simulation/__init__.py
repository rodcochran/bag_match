"""
Simulation package for running experiments and comparing different similarity measures.
"""

from .runner import run_simulation
from .config import ModelConfig, SimulationConfig
from .utils import (
    get_similarity_measures,
    print_bag_statistics,
    print_example_comparison,
    print_agreement_results,
    print_top_similar_bags
)
from bag_match.data.simulator import BagSimulator

__all__ = [
    'run_simulation',
    'ModelConfig',
    'SimulationConfig',
    'BagSimulator',
    'get_similarity_measures',
    'print_bag_statistics',
    'print_example_comparison',
    'print_agreement_results',
    'print_top_similar_bags'
] 
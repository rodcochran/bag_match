from bag_match.core import (
    BagSimilarity,
    find_top_similar_bags,
    get_similarity_measures,
    print_bag_statistics,
    print_example_comparison,
    print_agreement_results,
    print_top_similar_bags
)
from bag_match.data import BagSimulator
from bag_match.config import (
    ModelConfig,
    SimulationConfig,
    PreTrainedModelConfig,
    DEFAULT_PRETRAINED_CONFIG
)
from rich import print
from typing import List, Set, Dict, Tuple
import numpy as np
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import argparse
from bag_match.core.simulation_utils import parse_arguments, create_configurations, run_simulation


def main():
    # Parse arguments and create configurations
    args = parse_arguments()
    model_config, sim_config = create_configurations(args)
    
    # Run the simulation
    run_simulation(model_config, sim_config)

if __name__ == "__main__":
    main() 
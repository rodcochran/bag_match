from typing import List, Set, Dict, Tuple
import argparse
from bag_match.core.analysis import find_top_similar_bags
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from ..config import (
    ModelConfig,
    SimulationConfig,
    PreTrainedModelConfig,
    DEFAULT_PRETRAINED_CONFIG
)
from .similarity import BagSimilarity
from .utils import (
    get_similarity_measures,
    print_bag_statistics,
    print_example_comparison,
    print_agreement_results,
    print_top_similar_bags
)
from ..data import BagSimulator

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the simulation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Compare similarity measures for bags of items.')
    parser.add_argument('--num-bags', type=int, default=100,
                        help='Number of random bags to generate (default: 100)')
    parser.add_argument('--min-size', type=int, default=15,
                        help='Minimum size of each bag (default: 15)')
    parser.add_argument('--max-size', type=int, default=200,
                        help='Maximum size of each bag (default: 200)')
    parser.add_argument('--show-examples', type=int, default=3,
                        help='Number of example bags to show detailed results for (default: 3)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top similar bags to find for each bag (default: 10)')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=list(DEFAULT_PRETRAINED_CONFIG.models.keys()),
                        help='List of pre-trained models to use (default: all available models)')
    return parser.parse_args()

def create_configurations(args: argparse.Namespace) -> Tuple[ModelConfig, SimulationConfig]:
    """
    Create model and simulation configurations from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of ModelConfig and SimulationConfig
    """
    model_config = ModelConfig(
        model_names=args.models,
        embedding_cache_size=1000,
        text_cache_size=1024
    )
    
    sim_config = SimulationConfig(
        num_bags=args.num_bags,
        min_bag_size=args.min_size,
        max_bag_size=args.max_size,
        show_examples=args.show_examples,
        top_k=args.top_k
    )
    
    return model_config, sim_config

def run_simulation(model_config: ModelConfig, sim_config: SimulationConfig) -> None:
    """
    Run the bag similarity simulation with the given configurations.
    
    Args:
        model_config: Configuration for the models
        sim_config: Configuration for the simulation
    """
    # Initialize components
    similarity = BagSimilarity(model_config=model_config)
    simulator = BagSimulator(config=sim_config)
    
    # Generate random bags
    print(f"Generating {sim_config.num_bags} random bags...")
    bags = simulator.generate_bags()
    
    # Print bag statistics
    stats = simulator.get_bag_statistics(bags)
    print_bag_statistics(stats)
    
    # Get similarity measures
    similarity_measures = get_similarity_measures(similarity)
    
    # Find top matches using each similarity measure
    results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        for name, func in similarity_measures.items():
            task = progress.add_task(f"Calculating {name} similarities...", total=sim_config.num_bags)
            results[name] = find_top_similar_bags(
                bags, func, name, progress, task, sim_config.top_k
            )
    
    # Print results for example bags
    print(f"\nExample results for first {sim_config.show_examples} bags:")
    for i in range(min(sim_config.show_examples, sim_config.num_bags)):
        print_example_comparison(
            bags[i], bags[i+1], i, i+1, similarity, similarity_measures
        )
    
    # Print agreement results
    print_agreement_results(results, sim_config.top_k, "first bag")
    
    # Print top similar bags
    print_top_similar_bags(bags, results, sim_config.top_k) 
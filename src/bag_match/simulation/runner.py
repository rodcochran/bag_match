"""
Module for running bag similarity simulations.
"""

from typing import List, Set, Dict, Tuple
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from bag_match import BagMatcher
from bag_match.data.simulator import BagSimulator
from .config import ModelConfig, SimulationConfig
from .utils import (
    get_similarity_measures,
    print_bag_statistics,
    print_example_comparison,
    print_agreement_results,
    print_top_similar_bags
)

def run_simulation(model_config: ModelConfig, sim_config: SimulationConfig) -> None:
    """
    Run the bag similarity simulation with the given configurations.
    
    Args:
        model_config: Configuration for the models
        sim_config: Configuration for the simulation
    """
    # Initialize components
    matcher = BagMatcher(model_name=model_config.model_name)
    simulator = BagSimulator(config=sim_config)
    
    # Generate random bags
    print(f"Generating {sim_config.num_bags} random bags...")
    bags = simulator.generate_bags()
    
    # Print bag statistics
    stats = simulator.get_bag_statistics(bags)
    print_bag_statistics(stats)
    
    # Get similarity measures
    similarity_measures = get_similarity_measures(matcher)
    
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
            results[name] = []
            for i, bag in enumerate(bags):
                similar_bags = matcher.find_similar_bags(
                    query_bag=bag,
                    candidate_bags=bags[:i] + bags[i+1:],
                    top_k=sim_config.top_k,
                    similarity_method=name
                )
                results[name].append(similar_bags)
                progress.update(task, advance=1)
    
    # Print results for example bags
    print(f"\nExample results for first {sim_config.show_examples} bags:")
    for i in range(min(sim_config.show_examples, sim_config.num_bags)):
        print_example_comparison(
            bags[i], bags[i+1], i, i+1, matcher, similarity_measures
        )
    
    # Print agreement results
    print_agreement_results(results, sim_config.top_k, "first bag")
    
    # Print top similar bags
    print_top_similar_bags(bags, results, sim_config.top_k) 
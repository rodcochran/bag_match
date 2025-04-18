"""
Complex example demonstrating advanced usage of BagMatcher with the simulation framework.

This example shows how to:
1. Use the simulation framework to generate and analyze large bags
2. Compare different similarity measures
3. Analyze agreement between different methods
4. Generate detailed statistics and comparisons
"""

from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from bag_match import BagMatcher
from bag_match.simulation import (
    run_simulation,
    ModelConfig,
    SimulationConfig,
    print_bag_statistics,
    print_example_comparison,
    print_agreement_results,
    print_top_similar_bags
)
from bag_match.data.simulator import BagSimulator
import time

def get_similarity_measures(matcher: BagMatcher):
    """Get a dictionary of similarity measure functions."""
    return {
        # "jaccard": lambda b1, b2: matcher.compare_bags(b1, b2, "jaccard"),
        "embedding": lambda b1, b2: matcher.compare_bags(b1, b2, "embedding")
    }

def main():
    # Create configurations for different scenarios
    configs = [
        # Small bags with high similarity
        SimulationConfig(
            num_bags=10,
            min_bag_size=100,
            max_bag_size=200,
            show_examples=3,
            top_k=5
        ),
        # Medium bags with medium similarity
        SimulationConfig(
            num_bags=100,
            min_bag_size=100,
            max_bag_size=250,
            show_examples=3,
            top_k=5
        ),
        # Large bags with low similarity
        SimulationConfig(
            num_bags=10000,
            min_bag_size=50,
            max_bag_size=400,
            show_examples=10,
            top_k=25
        )
    ]

    configs = configs[2:]
    
    # Try different models
    models = [
        ModelConfig(model_name="all-MiniLM-L6-v2"),
        ModelConfig(model_name="all-mpnet-base-v2"),
        ModelConfig(model_name="multi-qa-mpnet-base-dot-v1")
    ]
    
    # Run simulations for each configuration and model
    for sim_config in configs:
        print(f"\n{'='*80}")
        print(f"Running simulation with configuration:")
        print(f"Number of bags: {sim_config.num_bags}")
        print(f"Bag size range: {sim_config.min_bag_size}-{sim_config.max_bag_size}")
        print(f"{'='*80}\n")
        
        for model_config in models:
            print(f"\nUsing model: {model_config.model_name}")
            start_time = time.time()
            
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
            for name, func in similarity_measures.items():
                print(f"\nCalculating {name} similarities...")
                results[name] = []
                with Progress() as progress:
                    task = progress.add_task(f"Processing bags...", total=len(bags))
                    for i, bag in enumerate(bags):
                        progress.update(task, advance=1)
                    similar_bags = matcher.find_similar_bags(
                        query_bag=bag,
                        candidate_bags=bags[:i] + bags[i+1:],
                        top_k=sim_config.top_k,
                        method=name
                    )
                    results[name].append(similar_bags)
            
            # Print results for example bags
            if sim_config.show_examples > 0:

                console = Console()
                table = Table(title=f"Example Results for First {sim_config.show_examples} Bags", show_header=True, header_style="bold")
                table.add_column("Bag Pair", style="cyan")
                table.add_column("Similarity Measure", style="cyan")
                table.add_column("Score", justify="right", style="green")

                for i in range(min(sim_config.show_examples, sim_config.num_bags)):
                    for name, func in similarity_measures.items():
                        similarity = func(bags[i], bags[i+1])
                        table.add_row(
                            f"Bags {i} & {i+1}",
                            name,
                            f"{similarity:.3f}"
                        )
                    if i < min(sim_config.show_examples, sim_config.num_bags) - 1:
                        table.add_row("", "", "")

                console.print(table)

            # Print agreement results
            print_agreement_results(results, sim_config.top_k, "first bag")
            
            # Print top similar bags
            print_top_similar_bags(bags, results, sim_config.top_k)
            
            elapsed_time = time.time() - start_time
            print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
            print(f"{'-'*80}\n")

if __name__ == "__main__":
    main() 
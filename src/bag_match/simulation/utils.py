"""
Utility functions for the simulation package.
"""

from typing import List, Set, Dict, Tuple, Callable
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from bag_match.core import BagMatcher

def get_similarity_measures(matcher: BagMatcher) -> Dict[str, Callable]:
    """
    Get a dictionary of similarity measure functions.
    
    Args:
        matcher: The BagMatcher instance to use
        
    Returns:
        Dictionary mapping measure names to functions
    """
    return {
        "cosine": lambda b1, b2: matcher.compare_bags(b1, b2, "cosine"),
        "jaccard": lambda b1, b2: matcher.compare_bags(b1, b2, "jaccard"),
        "pairwise": lambda b1, b2: matcher.compare_bags(b1, b2, "pairwise"),
        "concatenated": lambda b1, b2: matcher.compare_bags(b1, b2, "concatenated"),
        "pre-trained": lambda b1, b2: matcher.compare_bags(b1, b2, "pre-trained")
    }

def print_bag_statistics(stats: Dict) -> None:
    """
    Print statistics about the generated bags.
    
    Args:
        stats: Dictionary containing bag statistics
    """
    console = Console()
    table = Table(title="Bag Statistics", show_header=True, header_style="bold")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total bags", str(stats['num_bags']))
    table.add_row("Average size", f"{stats['avg_size']:.2f}")
    table.add_row("Min size", str(stats['min_size']))
    table.add_row("Max size", str(stats['max_size']))
    table.add_row("Total unique items", str(stats['total_unique_items']))
    table.add_row("Average unique items per bag", f"{stats['avg_unique_items_per_bag']:.2f}")

    console.print(table)

def print_example_comparison(
    bag1: Set[str],
    bag2: Set[str],
    idx1: int,
    idx2: int,
    matcher: BagMatcher,
    similarity_measures: Dict[str, Callable]
) -> None:
    """
    Print detailed comparison results for example bags.
    
    Args:
        bag1: First bag
        bag2: Second bag
        idx1: Index of first bag
        idx2: Index of second bag
        matcher: The BagMatcher instance
        similarity_measures: Dictionary of similarity measures
    """
    rprint(f"\nComparing bags {idx1} and {idx2}:")
    # rprint(f"Bag {idx1}: {bag1}")
    # rprint(f"Bag {idx2}: {bag2}")
    
    for name, func in similarity_measures.items():
        similarity = func(bag1, bag2)
        rprint(f"{name} similarity: {similarity:.3f}")

def calculate_agreement(
    results1: List[List[Tuple[Set[str], float]]],
    results2: List[List[Tuple[Set[str], float]]],
    top_k: int
) -> float:
    """
    Calculate agreement between two sets of results.
    
    Args:
        results1: First set of results
        results2: Second set of results
        top_k: Number of top similar bags considered
        
    Returns:
        Agreement score between 0 and 1
    """
    total = 0
    matches = 0
    
    for r1, r2 in zip(results1, results2):
        # Convert bags to frozenset to make them hashable
        top1 = {frozenset(bag) for bag, _ in r1[:top_k]}
        top2 = {frozenset(bag) for bag, _ in r2[:top_k]}
        matches += len(top1.intersection(top2))
        total += top_k
    
    return matches / total if total > 0 else 0.0

def print_agreement_results(
    results: Dict[str, List[List[Tuple[Set[str], float]]]],
    top_k: int,
    reference: str
) -> None:
    """
    Print agreement results between different similarity measures.
    
    Args:
        results: Dictionary of results for each similarity measure
        top_k: Number of top similar bags considered
        reference: Reference point for comparison
    """
    console = Console()
    table = Table(title="Agreement between Similarity Measures", show_header=True, header_style="bold")
    table.add_column("Measure 1", style="cyan")
    table.add_column("Measure 2", style="cyan") 
    table.add_column("Agreement", justify="right", style="green")

    measures = list(results.keys())
    for i, m1 in enumerate(measures):
        for m2 in measures[i+1:]:
            agreement = calculate_agreement(results[m1], results[m2], top_k)
            table.add_row(m1, m2, f"{agreement:.2%}")

    console.print(table)

def print_top_similar_bags(
    bags: List[Set[str]],
    results: Dict[str, List[List[Tuple[Set[str], float]]]],
    top_k: int
) -> None:
    """
    Print the top similar bags for each measure.
    
    Args:
        bags: List of all bags
        results: Dictionary of results for each similarity measure
        top_k: Number of top similar bags to show
    """
    console = Console()
    
    for measure, measure_results in results.items():
        table = Table(title=f"\nTop Similar Bags - {measure} Similarity", show_header=True, header_style="bold")
        table.add_column("Query Bag", style="cyan", justify="center")
        table.add_column("Similar Bag", style="cyan", justify="center")
        table.add_column("Similarity Score", justify="right", style="green")

        # Show first 3 query bags
        for i, bag_results in enumerate(measure_results[:3]):
            # Add results for this query bag
            for j, (bag, score) in enumerate(bag_results[:top_k]):
                # Find index of the similar bag in original bags list
                similar_bag_idx = bags.index(bag)
                table.add_row(
                    f"Bag {i}",
                    f"Bag {similar_bag_idx}",
                    f"{score:.3f}"
                )
            # Add separator between query bags
            if i < 2:  # Don't add after last bag
                table.add_row("", "", "")

        console.print(table)
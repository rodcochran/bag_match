from typing import List, Set, Dict, Tuple, Callable
from rich import print
from rich.progress import Progress
from .similarity import BagSimilarity
from .analysis import calculate_pairwise_similarities, calculate_agreement
from ..config import PreTrainedModelConfig, DEFAULT_PRETRAINED_CONFIG

def get_similarity_measures(similarity: BagSimilarity) -> Dict[str, Callable]:
    """
    Get a dictionary of similarity measures to compare.
    
    Args:
        similarity: BagSimilarity instance
        
    Returns:
        Dictionary mapping similarity measure names to their functions
    """
    measures = {
        "Jaccard": similarity.jaccard_similarity,
        "Average Pairwise": similarity.average_pairwise_similarity,
        "Concatenated": similarity.concatenated_similarity,
    }
    
    # Add pre-trained model measures if models are specified
    if similarity.model_config and similarity.model_config.model_names:
        for model_name in similarity.model_config.model_names:
            measures[f"Pre-trained ({model_name})"] = lambda b1, b2, m=model_name: similarity.pre_trained_similarity(b1, b2, m)
    
    return measures

def print_bag_statistics(stats: Dict[str, float]) -> None:
    """
    Print statistics about the generated bags.
    
    Args:
        stats: Dictionary containing bag statistics
    """
    print(f"\nBag statistics:")
    print(f"Number of bags: {stats['num_bags']}")
    print(f"Min size: {stats['min_size']}")
    print(f"Max size: {stats['max_size']}")
    print(f"Average size: {stats['avg_size']:.1f}")
    print(f"Total unique items: {stats['total_unique_items']}")
    print(f"Average unique items per bag: {stats['avg_unique_items_per_bag']:.1f}")

def print_example_comparison(
    bag1: Set[str],
    bag2: Set[str],
    bag1_idx: int,
    bag2_idx: int,
    similarity: BagSimilarity,
    similarity_measures: Dict[str, Callable]
) -> None:
    """
    Print detailed comparison of two example bags.
    
    Args:
        bag1: First bag of items
        bag2: Second bag of items
        bag1_idx: Index of first bag
        bag2_idx: Index of second bag
        similarity: BagSimilarity instance
        similarity_measures: Dictionary of similarity measures
    """
    print(f"\nComparing Bag {bag1_idx+1} (size: {len(bag1)} items) with Bag {bag2_idx+1} (size: {len(bag2)} items)")
    
    print(f"\n    Sample of items from Bag {bag1_idx+1}:")
    print("    " + str(sorted(list(bag1))[:10]))
    print(f"    Sample of items from Bag {bag2_idx+1}:")
    print("    " + str(sorted(list(bag2))[:10]))
    
    # Calculate and print individual measures
    results = calculate_pairwise_similarities(bag1, bag2, similarity_measures)
    
    # Print pre-trained model comparisons if models are specified
    if similarity.model_config and similarity.model_config.model_names:
        print(f"\n    Pre-trained model comparisons between Bag {bag1_idx+1} and Bag {bag2_idx+1}:")
        print("    " + "-" * 100)
        print("    " + f"{'Model':<40} {'Description':<40} {'Score':>8} {'Time (s)':>10}")
        print("    " + "-" * 100)
        for model_name in similarity.model_config.model_names:
            score, time_taken = results[f"Pre-trained ({model_name})"]
            print("    " + f"{model_name:<40} {DEFAULT_PRETRAINED_CONFIG.models.get(model_name, 'N/A'):<40} {score:>8.2f} {time_taken:>10.4f}")
        print("    " + "-" * 100)
    
    # Print final comparison
    print("\n    Comparison of all similarity measures:")
    print("    " + "-" * 120)
    print("    " + f"{'Method':<60} {'Score':>20} {'Time (s)':>20}")
    print("    " + "-" * 120)
    for name, (score, time_taken) in results.items():
        print("    " + f"{name:<60} {score:>20.2f} {time_taken:>20.4f}")
    print("    " + "-" * 120)

def print_agreement_results(
    results: Dict[str, List[List[Tuple[int, float]]]],
    top_k: int,
    bag_name: str
) -> None:
    """
    Print agreement between different similarity measures.
    
    Args:
        results: Dictionary mapping similarity measure names to their results
        top_k: Number of top similar bags considered
        bag_name: Name of the bag being analyzed
    """
    print(f"\nAgreement between similarity measures (for {bag_name}):")
    print("-" * 130)
    print(f"{'Measure 1':<55} {'Measure 2':<55} {'Common Bags':<20}")
    print("-" * 130)
    
    agreements = calculate_agreement(results, top_k)
    for m1, m2, common in agreements:
        print(f"{m1:<55} {m2:<55} {common:>2}/{top_k}")

def print_top_similar_bags(
    bags: List[Set[str]],
    results: Dict[str, List[List[Tuple[int, float]]]],
    top_k: int
) -> None:
    """
    Print top-k most similar bags for each query bag.
    
    Args:
        bags: List of bags
        results: Dictionary mapping similarity measure names to their results
        top_k: Number of top similar bags to show
    """
    print("\nTop similar bags for each query bag:")
    for query_idx in range(len(bags)):
        print(f"\nQuery Bag: {query_idx + 1}, Top {top_k} Similar Bags:")
        print("-" * 160)
        print(f"{'Method':<80} {'Similar Bag':<40} {'Score':>20}")
        print("-" * 160)
        
        for measure, similar_bags in results.items():
            # Get only top k matches for this measure, skipping self-matches
            matches = [
                (idx, score) for idx, score in similar_bags[query_idx]
                if idx != query_idx
            ][:top_k]
            
            for match_idx, score in matches:
                print(f"{measure:<80} {f'Bag {match_idx + 1}':<40} {score:>20.2f}")
        print("-" * 160)
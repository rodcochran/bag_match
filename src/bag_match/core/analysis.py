from typing import List, Set, Tuple, Dict, Callable
from rich.progress import Progress
import numpy as np

def find_top_similar_bags(
    bags: List[Set[str]],
    similarity_func: Callable[[Set[str], Set[str]], float],
    similarity_name: str,
    progress: Progress,
    task_id: int,
    top_k: int = 10
) -> List[List[Tuple[int, float]]]:
    """
    Find the top-k most similar bags for each bag using the given similarity function.
    
    Args:
        bags: List of bags to compare
        similarity_func: Function to calculate similarity between two bags
        similarity_name: Name of the similarity measure
        progress: Progress bar object
        task_id: Task ID for the progress bar
        top_k: Number of top similar bags to find for each bag
        
    Returns:
        List of lists containing (bag_index, similarity_score) tuples for top-k matches
    """
    n_bags = len(bags)
    top_matches = []
    
    # Cache for similarity scores
    similarity_cache = {}
    
    for i in range(n_bags):
        similarities = []
        for j in range(n_bags):
            if i != j:  # Don't compare a bag with itself
                # Check cache first - use sorted tuple as key to ensure (i,j) and (j,i) map to same entry
                cache_key = tuple(sorted([i, j]))
                if cache_key not in similarity_cache:
                    score = similarity_func(bags[i], bags[j])
                    similarity_cache[cache_key] = score
                else:
                    score = similarity_cache[cache_key]
                similarities.append((j, score))
        
        # Sort by similarity score in descending order and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches.append(similarities[:top_k])
        
        # Update progress
        progress.update(task_id, advance=1)
    
    return top_matches

def calculate_agreement(
    results: Dict[str, List[List[Tuple[int, float]]]],
    top_k: int
) -> List[Tuple[str, str, int]]:
    """
    Calculate agreement between different similarity measures.
    
    Args:
        results: Dictionary mapping similarity measure names to their results
        top_k: Number of top similar bags considered
        
    Returns:
        List of tuples containing (measure1, measure2, number_of_common_bags)
    """
    measures = list(results.keys())
    agreements = []
    
    for i in range(len(measures)):
        for j in range(i + 1, len(measures)):
            m1, m2 = measures[i], measures[j]
            # Get indices of top matches for first bag
            top1 = set(idx for idx, _ in results[m1][0])
            top2 = set(idx for idx, _ in results[m2][0])
            # Calculate number of common bags
            common = len(top1.intersection(top2))
            agreements.append((m1, m2, common))
    
    return agreements

def calculate_pairwise_similarities(
    bag1: Set[str],
    bag2: Set[str],
    similarity_funcs: Dict[str, Callable[[Set[str], Set[str]], float]]
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate pairwise similarities between two bags using multiple similarity functions.
    
    Args:
        bag1: First bag of items
        bag2: Second bag of items
        similarity_funcs: Dictionary mapping similarity measure names to their functions
        
    Returns:
        Dictionary mapping similarity measure names to (score, time_taken) tuples
    """
    import time
    results = {}
    
    for name, func in similarity_funcs.items():
        start_time = time.time()
        score = func(bag1, bag2)
        elapsed_time = time.time() - start_time
        results[name] = (score, elapsed_time)
    
    return results 
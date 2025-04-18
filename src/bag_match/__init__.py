"""
Bag Match - A library for finding similar bags of words.

This library provides tools for comparing and finding similar bags of words
using various similarity measures and embedding techniques.

The main interface is the BagMatcher class, which provides a simple way to
compare bags of words and find similar bags from a collection of candidates.

Example:
    >>> from bag_match import BagMatcher
    >>> matcher = BagMatcher()
    >>> query_bag = {"apple", "banana", "orange"}
    >>> candidate_bags = [
    ...     {"apple", "pear", "grape"},
    ...     {"banana", "kiwi", "mango"},
    ...     {"orange", "lemon", "lime"}
    ... ]
    >>> similar_bags = matcher.find_similar_bags(query_bag, candidate_bags, top_k=2)
    >>> for bag, score in similar_bags:
    ...     print(f"Similarity: {score:.3f}, Bag: {bag}")
"""

from bag_match.core.matcher import BagMatcher

# Deprecated imports - will be removed in future versions
from bag_match.core.analysis import (
    find_top_similar_bags,
    get_similarity_measures,
    print_bag_statistics,
    print_example_comparison,
    print_agreement_results,
    print_top_similar_bags
)

__version__ = "0.1.0"
__all__ = [
    "BagMatcher",
    # Deprecated exports
    "find_top_similar_bags",
    "get_similarity_measures",
    "print_bag_statistics",
    "print_example_comparison",
    "print_agreement_results",
    "print_top_similar_bags"
] 
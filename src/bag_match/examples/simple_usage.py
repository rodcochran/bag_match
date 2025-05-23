"""
Simple example demonstrating the usage of BagMatcher.

This example shows how to:
1. Initialize a BagMatcher
2. Compare two bags directly
3. Find similar bags from a collection
"""

from rich import print
from bag_match import BagMatcher

def main():
    # Initialize the matcher with default settings
    matcher = BagMatcher(model_name="all-MiniLM-L6-v2")
    
    # Example bags
    query_bag = {"apple", "banana", "orange"}
    candidate_bags = [
        {"apple", "pear", "grape"},
        {"banana", "kiwi", "mango"},
        {"orange", "lemon", "lime"},
        {"apple", "banana", "pear"},
        {"orange", "grape", "kiwi"}
    ]
    
    # Compare two bags directly
    print("Comparing two bags:")
    similarity = matcher.compare_bags(
        bag1={"apple", "banana"},
        bag2={"apple", "pear"}
    )
    print(f"Similarity between {{'apple', 'banana'}} and {{'apple', 'pear'}}: {similarity:.3f}")
    print()
    
    # Find similar bags using embedding similarity
    print("Finding similar bags using embedding similarity:")
    similar_bags = matcher.find_similar_bags(
        query_bag=query_bag,
        candidate_bags=candidate_bags,
        top_k=3,
        method="embedding"
    )
    
    print(f"Top 3 similar bags to {query_bag}:")
    for bag, score in similar_bags:
        print(f"Similarity: {score:.3f}, Bag: {bag}")
    print()
    
    # Find similar bags using Jaccard similarity
    print("Finding similar bags using Jaccard similarity:")
    similar_bags = matcher.find_similar_bags(
        query_bag=query_bag,
        candidate_bags=candidate_bags,
        top_k=3,
        method="jaccard"
    )
    
    print(f"Top 3 similar bags to {query_bag}:")
    for bag, score in similar_bags:
        print(f"Similarity: {score:.3f}, Bag: {bag}")

if __name__ == "__main__":
    main() 
from typing import List, Set, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from functools import lru_cache

class BagMatcher:
    """
    A class for finding similar bags of words using various similarity measures.
    
    This class provides a simple interface for comparing bags of words and finding
    the most similar bags from a collection of candidate bags.
    
    Example:
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
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the BagMatcher with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use for embeddings.
                             Defaults to "all-MiniLM-L6-v2" which is a good balance of speed and quality.
                             Other options include:
                             - "all-mpnet-base-v2": Higher quality but slower
                             - "multi-qa-mpnet-base-dot-v1": Optimized for semantic search
                             - "paraphrase-multilingual-mpnet-base-v2": Multilingual support
        """
        self.model = SentenceTransformer(model_name, cache_folder="./models")
        self._embedding_cache: Dict[str, NDArray[np.float32]] = {}
        self.model.to("mps")
    
    @lru_cache(maxsize=1000)
    def _get_word_embedding(self, word: str) -> NDArray[np.float32]:
        """Get the embedding for a word, using caching to avoid recomputation."""
        if word not in self._embedding_cache:
            self._embedding_cache[word] = self.model.encode(word, convert_to_numpy=True)
        return self._embedding_cache[word]
    
    def _jaccard_similarity(self, bag1: Set[str], bag2: Set[str]) -> float:
        """Calculate the Jaccard similarity between two bags of words."""
        intersection = len(bag1.intersection(bag2))
        union = len(bag1.union(bag2))
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
        """Calculate the cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _get_bag_embedding(self, bag: Set[str]) -> NDArray[np.float32]:
        """Get the average embedding for a bag of words."""
        embeddings = [self._get_word_embedding(word) for word in bag]
        return np.mean(embeddings, axis=0)
    
    def compare_bags(
        self,
        bag1: Set[str],
        bag2: Set[str],
        similarity_method: str = "cosine"
    ) -> float:
        """
        Compare two bags of words using the specified similarity method.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            similarity_method (str): Method to use for comparison. Options:
                                   - "cosine": Uses average word embeddings (default)
                                   - "jaccard": Uses Jaccard similarity
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if similarity_method == "jaccard":
            return self._jaccard_similarity(bag1, bag2)
        elif similarity_method == "cosine":
            embedding1 = self._get_bag_embedding(bag1)
            embedding2 = self._get_bag_embedding(bag2)
            return self._cosine_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity method: {similarity_method}")
    
    def find_similar_bags(
        self,
        query_bag: Set[str],
        candidate_bags: List[Set[str]],
        top_k: int = 5,
        similarity_method: str = "cosine"
    ) -> List[Tuple[Set[str], float]]:
        """
        Find the top k most similar bags to the query bag.
        
        Args:
            query_bag (Set[str]): The bag to find similar bags for
            candidate_bags (List[Set[str]]): List of candidate bags to compare against
            top_k (int): Number of most similar bags to return
            similarity_method (str): Method to use for comparison (see compare_bags)
            
        Returns:
            List[Tuple[Set[str], float]]: List of (bag, similarity_score) tuples,
                                         sorted by similarity score in descending order
        """
        # Calculate similarities
        similarities = [
            (bag, self.compare_bags(query_bag, bag, similarity_method))
            for bag in candidate_bags
        ]
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k] 
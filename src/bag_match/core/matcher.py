"""
Core functionality for comparing bags of words.
"""

from typing import List, Set, Dict, Tuple, Optional
import torch
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
        >>> matcher = BagMatcher()  # Uses Jaccard similarity by default
        >>> query_bag = {"apple", "banana", "orange"}
        >>> candidate_bags = [
        ...     {"apple", "pear", "grape"},
        ...     {"banana", "kiwi", "mango"},
        ...     {"orange", "lemon", "lime"}
        ... ]
        >>> similar_bags = matcher.find_similar_bags(query_bag, candidate_bags, top_k=2)
        >>> for bag, score in similar_bags:
        ...     print(f"Similarity: {score:.3f}, Bag: {bag}")
        
        # With semantic similarity using a pre-trained model
        >>> matcher = BagMatcher(model_name="all-MiniLM-L6-v2")
        >>> similar_bags = matcher.find_similar_bags(query_bag, candidate_bags, method="embedding")
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the BagMatcher.
        
        Args:
            model_name (Optional[str]): Name of the sentence transformer model to use for embeddings.
                                      If None, only Jaccard similarity will be available.
                                      Common options:
                                      - "all-MiniLM-L6-v2": Good balance of speed and quality
                                      - "all-mpnet-base-v2": Higher quality but slower
        """
        self.model = None
        self._embedding_cache: Dict[str, NDArray[np.float32]] = {}
        
        if model_name is not None:
            self.model = SentenceTransformer(model_name, cache_folder="./models")
            # Use MPS if available on Mac, CUDA if available on GPU, otherwise CPU
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.model.to("mps")
            elif torch.cuda.is_available():
                self.model.to("cuda")
            else:
                self.model.to("cpu")
    
    @lru_cache(maxsize=1000)
    def _get_word_embedding(self, word: str) -> NDArray[np.float32]:
        """Get the embedding for a word, using caching to avoid recomputation."""
        if self.model is None:
            raise ValueError("Model not initialized. Please provide a model_name when creating BagMatcher.")
            
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
        method: str = "jaccard"
    ) -> float:
        """
        Compare two bags of words using the specified similarity method.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            method (str): Method to use for comparison. Options:
                         - "jaccard": Uses Jaccard similarity (default)
                         - "embedding": Uses average word embeddings (requires model)
            
        Returns:
            float: Similarity score between 0 and 1
            
        Raises:
            ValueError: If method is "embedding" but no model was provided
        """
        if method == "jaccard":
            return self._jaccard_similarity(bag1, bag2)
        elif method == "embedding":
            if self.model is None:
                raise ValueError("Embedding similarity requires a model. Please provide a model_name when creating BagMatcher.")
            embedding1 = self._get_bag_embedding(bag1)
            embedding2 = self._get_bag_embedding(bag2)
            return self._cosine_similarity(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity method: {method}. Use 'jaccard' or 'embedding'.")
    
    def find_similar_bags(
        self,
        query_bag: Set[str],
        candidate_bags: List[Set[str]],
        top_k: int = 5,
        method: str = "jaccard"
    ) -> List[Tuple[Set[str], float]]:
        """
        Find the top k most similar bags to the query bag.
        
        Args:
            query_bag (Set[str]): The bag to find similar bags for
            candidate_bags (List[Set[str]]): List of candidate bags to compare against
            top_k (int): Number of most similar bags to return
            method (str): Method to use for comparison (see compare_bags)
            
        Returns:
            List[Tuple[Set[str], float]]: List of (bag, similarity_score) tuples,
                                         sorted by similarity score in descending order
        """
        # Calculate similarities
        similarities = [
            (bag, self.compare_bags(query_bag, bag, method))
            for bag in candidate_bags
        ]
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    def pairwise_similarity_matrix(self, bag1: Set[str], bag2: Set[str]) -> NDArray[np.float32]:
        """
        Calculate the pairwise similarity matrix between two bags of words.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            
        Returns:
            NDArray[np.float32]: Matrix of shape (len(bag2), len(bag1)) containing
                                pairwise similarities between words in the bags
        """
        # Convert bags to lists for consistent ordering
        words1 = list(bag1)
        words2 = list(bag2)
        
        # Get embeddings for all words
        embeddings1 = np.array([self._get_word_embedding(word) for word in words1])
        embeddings2 = np.array([self._get_word_embedding(word) for word in words2])
        
        # Calculate pairwise similarities
        similarity_matrix = np.dot(embeddings2, embeddings1.T)
        
        # Normalize the similarity matrix
        norms1 = np.linalg.norm(embeddings1, axis=1)
        norms2 = np.linalg.norm(embeddings2, axis=1)
        similarity_matrix = similarity_matrix / (norms2[:, np.newaxis] * norms1)
        
        return similarity_matrix
    
    def average_pairwise_similarity(self, bag1: Set[str], bag2: Set[str]) -> float:
        """
        Calculate the average pairwise similarity between two bags of words,
        avoiding double counting of pairs.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            
        Returns:
            float: Average similarity score between -1 and 1
        """
        similarity_matrix = self.pairwise_similarity_matrix(bag1, bag2)
        
        # Get the upper triangle of the matrix (excluding diagonal)
        # This gives us only unique pairs
        upper_triangle = np.triu(similarity_matrix, k=1)
        
        # Count non-zero elements in upper triangle
        num_pairs = np.count_nonzero(upper_triangle)
        
        if num_pairs == 0:
            return 0.0
            
        # Calculate average of unique pairs
        return float(np.sum(upper_triangle) / num_pairs)
    
    def concatenated_similarity(self, bag1: Set[str], bag2: Set[str]) -> float:
        """
        Calculate the similarity between two bags of words by concatenating all words
        in each bag into a single string and computing their similarity.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        # Convert bags to sorted lists for consistent ordering
        words1 = sorted(list(bag1))
        words2 = sorted(list(bag2))
        
        # Concatenate words with spaces
        concat1 = " ".join(words1)
        concat2 = " ".join(words2)
        
        # Get embeddings for concatenated strings
        embedding1 = self._get_word_embedding(concat1)
        embedding2 = self._get_word_embedding(concat2)
        
        # Calculate cosine similarity
        return self._cosine_similarity(embedding1, embedding2)
    
    def pre_trained_similarity(self, bag1: Set[str], bag2: Set[str]) -> float:
        """
        Calculate similarity between two bags of words using the pre-trained model.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            
        Returns:
            float: Similarity score between -1 and 1
        """
        # Convert bags to sorted lists for consistent ordering
        words1 = sorted(list(bag1))
        words2 = sorted(list(bag2))
        
        # Create a natural language description of the bags
        desc1 = f"This is a collection of items: {', '.join(words1)}"
        desc2 = f"This is a collection of items: {', '.join(words2)}"
        
        # Get embeddings for the descriptions
        embedding1 = self._get_word_embedding(desc1)
        embedding2 = self._get_word_embedding(desc2)
        
        # Calculate cosine similarity
        return self._cosine_similarity(embedding1, embedding2)
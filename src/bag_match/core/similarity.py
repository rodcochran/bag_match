from typing import List, Set, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from ..config.settings import ModelConfig, PreTrainedModelConfig

class BagSimilarity:
    def __init__(self, model_config: Optional[ModelConfig] = None, 
                 cache_folder: str = "./models",
                 device: str = "mps",
                 embedding_cache_size: int = 1000,
                 text_cache_size: int = 1024):
        """
        Initialize the BagSimilarity class.
        
        Args:
            model_config: Configuration for the sentence transformer model
            cache_folder: Folder to cache model files
            device: Device to run the model on (mps, cuda, or cpu)
            embedding_cache_size: Size of the embedding cache
            text_cache_size: Size of the text cache
        """
        self.model_config = model_config or ModelConfig()
        self.cache_folder = cache_folder
        self.device = device
        self.embedding_cache_size = embedding_cache_size
        self.text_cache_size = text_cache_size
        
        # Initialize the default model if models are specified
        if self.model_config.model_names:
            self.model = SentenceTransformer(
                self.model_config.model_names[0],
                cache_folder=self.cache_folder
            )
            self.model.to(self.device)
        else:
            self.model = None
            
        # Initialize caches
        self.embedding_cache = {}
        self.text_cache = {}
        
    @lru_cache(maxsize=1000)
    def get_word_embedding(self, word: str) -> NDArray[np.float32]:
        """
        Get the embedding for a word, using caching to avoid recomputation.
        
        Args:
            word (str): The word to get the embedding for.
            
        Returns:
            NDArray[np.float32]: The embedding vector for the word.
        """
        if word not in self.embedding_cache:
            self.embedding_cache[word] = self.model.encode(word, convert_to_numpy=True)
        return self.embedding_cache[word]

    def jaccard_similarity(self, bag1: Set[str], bag2: Set[str]) -> float:
        """
        Calculate the Jaccard similarity between two bags of words.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            
        Returns:
            float: Jaccard similarity score between 0 and 1
        """
        intersection = len(bag1.intersection(bag2))
        union = len(bag1.union(bag2))
        return intersection / union if union > 0 else 0.0

    def cosine_similarity(self, vec1: NDArray[np.float32], vec2: NDArray[np.float32]) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Args:
            vec1 (NDArray[np.float32]): First vector
            vec2 (NDArray[np.float32]): Second vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

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
        embeddings1 = np.array([self.get_word_embedding(word) for word in words1])
        embeddings2 = np.array([self.get_word_embedding(word) for word in words2])
        
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

    def cosine_similarity_bags(self, bag1_embeddings: List[NDArray[np.float32]], 
                             bag2_embeddings: List[NDArray[np.float32]]) -> float:
        """
        Calculate the cosine similarity between two bags of word embeddings by averaging
        the embeddings and computing their cosine similarity.
        
        Args:
            bag1_embeddings (List[NDArray[np.float32]]): List of embeddings for the first bag
            bag2_embeddings (List[NDArray[np.float32]]): List of embeddings for the second bag
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        # Convert lists to numpy arrays
        bag1_embeddings = np.array(bag1_embeddings)
        bag2_embeddings = np.array(bag2_embeddings)
        
        # Calculate average embeddings for each bag
        avg_embedding1 = np.mean(bag1_embeddings, axis=0)
        avg_embedding2 = np.mean(bag2_embeddings, axis=0)
        
        # Calculate cosine similarity between the average embeddings
        return self.cosine_similarity(avg_embedding1, avg_embedding2)

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
        embedding1 = self.model.encode(concat1, convert_to_numpy=True)
        embedding2 = self.model.encode(concat2, convert_to_numpy=True)
        
        # Calculate cosine similarity
        return self.cosine_similarity(embedding1, embedding2)

    def pre_trained_similarity(self, bag1: Set[str], bag2: Set[str], 
                             model_name: Optional[str] = None) -> float:
        """
        Calculate similarity between two bags of words using a pre-trained model.
        
        Args:
            bag1 (Set[str]): First bag of words
            bag2 (Set[str]): Second bag of words
            model_name (str): Name of the pre-trained model to use
            
        Returns:
            float: Similarity score between -1 and 1
        """
        if model_name is None:
            model_name = self.model_config.model_names[0]
            
        # Create a temporary model instance for this specific model
        temp_model = SentenceTransformer(model_name, cache_folder=self.cache_folder)
        temp_model.to(self.device)
        
        # Convert bags to sorted lists for consistent ordering
        words1 = sorted(list(bag1))
        words2 = sorted(list(bag2))
        
        # Create a natural language description of the bags
        desc1 = f"This is a collection of items: {', '.join(words1)}"
        desc2 = f"This is a collection of items: {', '.join(words2)}"
        
        # Get embeddings for the descriptions, using caching
        @lru_cache(maxsize=self.text_cache_size)
        def get_cached_embedding(text: str) -> np.ndarray:
            return temp_model.encode(text, convert_to_numpy=True)
            
        embedding1 = get_cached_embedding(desc1)
        embedding2 = get_cached_embedding(desc2)
        
        # Calculate cosine similarity
        return self.cosine_similarity(embedding1, embedding2) 
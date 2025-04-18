# Bag Match

A Python library for finding similar bags of words using various similarity measures and embedding techniques.

## Installation

```bash
pip install bag-match
```

## Quick Start

```python
from bag_match import BagMatcher

# Initialize the matcher
matcher = BagMatcher()

# Your bags
query_bag = {"apple", "banana", "orange"}
candidate_bags = [
    {"apple", "pear", "grape"},
    {"banana", "kiwi", "mango"},
    {"orange", "lemon", "lime"}
]

# Find similar bags
similar_bags = matcher.find_similar_bags(
    query_bag=query_bag,
    candidate_bags=candidate_bags,
    top_k=2
)

# Print results
for bag, score in similar_bags:
    print(f"Similarity: {score:.3f}, Bag: {bag}")
```

## Features

- Simple interface for comparing bags of words
- Multiple similarity measures:
  - Cosine similarity using word embeddings
  - Jaccard similarity
- Efficient caching of word embeddings
- Support for different embedding models
- Easy to extend with custom similarity measures

## Usage

### Basic Usage

```python
from bag_match import BagMatcher

# Initialize with default settings
matcher = BagMatcher()

# Compare two bags directly
similarity = matcher.compare_bags(
    bag1={"apple", "banana"},
    bag2={"apple", "pear"}
)

# Find similar bags
similar_bags = matcher.find_similar_bags(
    query_bag={"apple", "banana", "orange"},
    candidate_bags=[
        {"apple", "pear", "grape"},
        {"banana", "kiwi", "mango"},
        {"orange", "lemon", "lime"}
    ],
    top_k=2
)
```

### Using Different Models

```python
# Initialize with a different model
matcher = BagMatcher(model_name="all-mpnet-base-v2")  # Higher quality but slower
```

### Using Different Similarity Measures

```python
# Use Jaccard similarity
similarity = matcher.compare_bags(
    bag1={"apple", "banana"},
    bag2={"apple", "pear"},
    similarity_method="jaccard"
)

# Find similar bags using Jaccard similarity
similar_bags = matcher.find_similar_bags(
    query_bag={"apple", "banana", "orange"},
    candidate_bags=[...],
    similarity_method="jaccard"
)
```

## Running Simulations

The library includes a simulation package for running experiments and comparing different similarity measures. The simulation uses a rich set of real words (fruits, vegetables, etc.) to generate interesting and realistic comparisons.

### Running a Simulation

```python
from bag_match.simulation import (
    run_simulation,
    ModelConfig,
    SimulationConfig
)

# Create configurations
model_config = ModelConfig(
    model_name="all-MiniLM-L6-v2"  # Good balance of speed and quality
)

sim_config = SimulationConfig(
    num_bags=100,        # Number of random bags to generate
    min_bag_size=15,     # Minimum size of each bag
    max_bag_size=200,    # Maximum size of each bag
    show_examples=3,     # Number of example bags to show detailed results for
    top_k=10            # Number of top similar bags to find for each bag
)

# Run the simulation
run_simulation(model_config, sim_config)
```

The simulation will:
1. Generate random bags of items (using a rich set of real words)
2. Calculate similarities using different measures
3. Show statistics about the bags
4. Display example comparisons
5. Show agreement between different similarity measures
6. Display the top similar bags for each measure

## API Reference

### BagMatcher

The main class for comparing bags of words.

#### `__init__(self, model_name: str = "all-MiniLM-L6-v2")`

Initialize the BagMatcher with a sentence transformer model.

- `model_name`: Name of the sentence transformer model to use for embeddings.
  - `"all-MiniLM-L6-v2"`: Good balance of speed and quality (default)
  - `"all-mpnet-base-v2"`: Higher quality but slower
  - `"multi-qa-mpnet-base-dot-v1"`: Optimized for semantic search
  - `"paraphrase-multilingual-mpnet-base-v2"`: Multilingual support

#### `compare_bags(self, bag1: Set[str], bag2: Set[str], similarity_method: str = "cosine") -> float`

Compare two bags of words using the specified similarity method.

- `bag1`: First bag of words
- `bag2`: Second bag of words
- `similarity_method`: Method to use for comparison
  - `"cosine"`: Uses average word embeddings (default)
  - `"jaccard"`: Uses Jaccard similarity
- Returns: Similarity score between 0 and 1

#### `find_similar_bags(self, query_bag: Set[str], candidate_bags: List[Set[str]], top_k: int = 5, similarity_method: str = "cosine") -> List[Tuple[Set[str], float]]`

Find the top k most similar bags to the query bag.

- `query_bag`: The bag to find similar bags for
- `candidate_bags`: List of candidate bags to compare against
- `top_k`: Number of most similar bags to return
- `similarity_method`: Method to use for comparison
- Returns: List of (bag, similarity_score) tuples, sorted by similarity score in descending order

## License

MIT 
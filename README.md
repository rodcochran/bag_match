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
- Advanced simulation framework for large-scale experiments
- Rich visualization of results and statistics
- Progress tracking for long-running operations

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

## Advanced Usage

### Running Large-Scale Simulations

The library includes a comprehensive simulation framework for running experiments and comparing different similarity measures. You can configure various aspects of the simulation including bag sizes, number of bags, and similarity measures to use.

```python
from bag_match import BagMatcher
from bag_match.simulation import (
    ModelConfig,
    SimulationConfig,
    BagSimulator
)
from rich.progress import Progress

# Create simulation configuration
sim_config = SimulationConfig(
    num_bags=10000,      # Number of random bags to generate
    min_bag_size=50,     # Minimum size of each bag
    max_bag_size=400,    # Maximum size of each bag
    show_examples=10,    # Number of example bags to show detailed results for
    top_k=25            # Number of top similar bags to find for each bag
)

# Initialize components
matcher = BagMatcher(model_name="all-mpnet-base-v2")
simulator = BagSimulator(config=sim_config)

# Generate random bags
print(f"Generating {sim_config.num_bags} random bags...")
bags = simulator.generate_bags()

# Calculate similarities with progress tracking
with Progress() as progress:
    task = progress.add_task("Processing bags...", total=len(bags))
    for i, bag in enumerate(bags):
        progress.update(task, advance=1)
        similar_bags = matcher.find_similar_bags(
            query_bag=bag,
            candidate_bags=bags[:i] + bags[i+1:],
            top_k=sim_config.top_k
        )
```

### Comparing Multiple Models

You can easily compare the performance of different embedding models:

```python
models = [
    ModelConfig(model_name="all-MiniLM-L6-v2"),    # Fast and efficient
    ModelConfig(model_name="all-mpnet-base-v2"),   # High quality
    ModelConfig(model_name="multi-qa-mpnet-base-dot-v1")  # Optimized for search
]

for model_config in models:
    matcher = BagMatcher(model_name=model_config.model_name)
    # Run your analysis with each model
```

### Custom Similarity Measures

You can define and use custom similarity measures:

```python
def get_similarity_measures(matcher: BagMatcher):
    """Get a dictionary of similarity measure functions."""
    return {
        "embedding": lambda b1, b2: matcher.compare_bags(b1, b2, "embedding"),
        "jaccard": lambda b1, b2: matcher.compare_bags(b1, b2, "jaccard")
    }

# Use the custom measures
similarity_measures = get_similarity_measures(matcher)
for name, func in similarity_measures.items():
    similarity = func(bag1, bag2)
```

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

### Simulation Framework

The simulation framework provides tools for running large-scale experiments and analyzing results.

#### `SimulationConfig`

Configuration for running simulations.

- `num_bags`: Number of random bags to generate
- `min_bag_size`: Minimum size of each bag
- `max_bag_size`: Maximum size of each bag
- `show_examples`: Number of example bags to show detailed results for
- `top_k`: Number of top similar bags to find for each bag

#### `ModelConfig`

Configuration for different embedding models.

- `model_name`: Name of the sentence transformer model to use

#### `BagSimulator`

Class for generating and analyzing random bags of words.

- `generate_bags()`: Generate random bags according to the configuration
- `get_bag_statistics(bags)`: Get statistics about the generated bags

## License

MIT 
"""
Example script demonstrating how to run a bag similarity simulation.
"""

from rich import print
from bag_match.simulation import (
    run_simulation,
    ModelConfig,
    SimulationConfig
)

def main():
    # Create configurations
    model_config = ModelConfig(
        model_name="all-MiniLM-L6-v2"
    )
    
    sim_config = SimulationConfig(
        num_bags=10,
        min_bag_size=50,
        max_bag_size=100,
        show_examples=3,
        top_k=5
    )
    
    # Run the simulation
    run_simulation(model_config, sim_config)

if __name__ == "__main__":
    main() 
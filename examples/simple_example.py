#!/usr/bin/env python3
"""
Simple example: Train a small model with QLoRA for quick testing
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.memory_tracker import MemoryTracker
from utils.cost_calculator import CostCalculator

def main():
    print("="*60)
    print("Budget Finetuning Pipeline - Simple Example")
    print("="*60)
    
    # Initialize utilities
    print("\n1. Initializing memory tracker...")
    memory_tracker = MemoryTracker()
    memory_tracker.print_memory_stats()
    
    print("\n2. Initializing cost calculator...")
    cost_calculator = CostCalculator(gpu_type="A100_40GB")
    print(f"   GPU Type: {cost_calculator.gpu_type}")
    print(f"   Hourly Rate: ${cost_calculator.hourly_rate}")
    
    # Simulate training metrics
    print("\n3. Simulating training scenario...")
    training_time = 2.5 * 3600  # 2.5 hours in seconds
    peak_memory = 18.4  # GB
    method = "qlora"
    num_epochs = 3
    
    # Calculate costs
    print("\n4. Calculating costs...")
    cost_per_epoch = cost_calculator.calculate_training_cost(
        training_time / num_epochs,
        peak_memory,
        method
    )
    total_cost = cost_per_epoch * num_epochs
    
    # Estimate savings
    baseline_cost = 40.0  # Full finetuning estimate
    savings = cost_calculator.calculate_savings(baseline_cost, total_cost)
    
    # Print results
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Training Method: {method.upper()}")
    print(f"Training Time: {training_time/3600:.2f} hours")
    print(f"Peak Memory: {peak_memory:.1f} GB")
    print(f"Cost per Epoch: ${cost_per_epoch:.2f}")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"\nSavings vs Full Finetuning:")
    print(f"  Absolute: ${savings['absolute_savings']:.2f}")
    print(f"  Percentage: {savings['percent_savings']:.1f}%")
    print("="*60)
    
    # Run actual training
    print("\n5. To run actual training, use:")
    print("\n   python src/training/train_qlora.py \\")
    print("       --model_name meta-llama/Llama-2-7b-hf \\")
    print("       --num_epochs 3 \\")
    print("       --output_dir ./results/qlora_test")
    print("\n   Or use the benchmark runner:")
    print("\n   python run_benchmark.py --methods qlora --num_epochs 1")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

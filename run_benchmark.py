#!/usr/bin/env python3
"""
Benchmark runner script
Runs both QLoRA and Spectrum finetuning and compares results
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run finetuning benchmarks")
    parser.add_argument("--methods", nargs="+", 
                       choices=["qlora", "spectrum", "all"],
                       default=["all"],
                       help="Methods to benchmark")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Base model to use")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--skip_visualization", action="store_true",
                       help="Skip generating visualizations")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip model evaluation")
    return parser.parse_args()


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error running {description}")
        return False
    
    return True


def run_qlora_training(args):
    """Run QLoRA training"""
    cmd = [
        sys.executable,
        "src/training/train_qlora.py",
        "--model_name", args.model_name,
        "--num_epochs", str(args.num_epochs),
        "--output_dir", "./results/qlora",
    ]
    
    return run_command(cmd, "QLoRA Training")


def run_spectrum_training(args):
    """Run Spectrum training"""
    cmd = [
        sys.executable,
        "src/training/train_spectrum.py",
        "--model_name", args.model_name,
        "--num_epochs", str(args.num_epochs),
        "--output_dir", "./results/spectrum",
    ]
    
    return run_command(cmd, "Spectrum Training")


def run_evaluation(model_path, output_dir, base_model):
    """Run model evaluation"""
    cmd = [
        sys.executable,
        "src/evaluation/evaluate_model.py",
        "--model_path", model_path,
        "--base_model", base_model,
        "--output_dir", output_dir,
        "--num_samples", "500",
    ]
    
    return run_command(cmd, f"Evaluation for {model_path}")


def run_visualization():
    """Generate comparison visualizations"""
    cmd = [
        sys.executable,
        "src/evaluation/visualize_results.py",
        "--results_dir", "./results",
        "--output_dir", "./results/plots",
    ]
    
    return run_command(cmd, "Visualization Generation")


def main():
    args = parse_args()
    
    print("="*60)
    print("Budget Finetuning Pipeline Benchmark")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Methods: {args.methods}")
    print("="*60)
    
    # Determine which methods to run
    methods = args.methods
    if "all" in methods:
        methods = ["qlora", "spectrum"]
    
    # Track results
    results = {}
    
    # Run training for each method
    for method in methods:
        print(f"\n\nStarting {method.upper()} training...")
        
        if method == "qlora":
            success = run_qlora_training(args)
        elif method == "spectrum":
            success = run_spectrum_training(args)
        else:
            print(f"Unknown method: {method}")
            continue
        
        results[method] = success
        
        if success and not args.skip_evaluation:
            # Run evaluation
            model_path = f"./results/{method}/final_model"
            eval_output = f"./results/{method}/evaluation"
            
            if Path(model_path).exists():
                print(f"\nEvaluating {method} model...")
                run_evaluation(model_path, eval_output, args.model_name)
    
    # Generate visualizations
    if not args.skip_visualization and any(results.values()):
        print("\n\nGenerating comparison visualizations...")
        run_visualization()
    
    # Summary
    print("\n\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)
    
    for method, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"{method.upper()}: {status}")
    
    print("\nResults saved to: ./results/")
    print("Visualizations saved to: ./results/plots/")
    print("="*60)


if __name__ == "__main__":
    main()

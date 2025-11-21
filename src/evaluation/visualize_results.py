"""
Visualization script for comparing training methods
Generates the 'Finetuning on a Budget' comparison graphs
"""

import os
import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison visualizations")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory containing training results")
    parser.add_argument("--output_dir", type=str, default="./results/plots",
                        help="Output directory for plots")
    return parser.parse_args()


def load_metrics(results_dir):
    """Load metrics from all training runs"""
    metrics = {}
    
    for method_dir in Path(results_dir).iterdir():
        if method_dir.is_dir():
            metrics_file = method_dir / "metrics" / "training_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    method_name = method_dir.name
                    metrics[method_name] = json.load(f)
    
    return metrics


def plot_cost_comparison(metrics, output_dir):
    """Plot cost comparison between methods"""
    methods = []
    costs = []
    
    for method, data in metrics.items():
        methods.append(method.upper())
        costs.append(data.get("total_cost_usd", 0))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(methods, costs, color=colors[:len(methods)])
    
    # Customize
    plt.title("Finetuning on a Budget: Cost Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Total Cost (USD)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'cost_comparison.png')}")
    plt.close()


def plot_memory_comparison(metrics, output_dir):
    """Plot memory usage comparison"""
    methods = []
    memory_usage = []
    
    for method, data in metrics.items():
        methods.append(method.upper())
        memory_usage.append(data.get("peak_memory_gb", 0))
    
    # Create figure
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(methods, memory_usage, color=colors[:len(methods)])
    
    # Customize
    plt.title("VRAM Footprint Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Peak VRAM (GB)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mem in zip(bars, memory_usage):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f} GB',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'memory_comparison.png')}")
    plt.close()


def plot_training_time_comparison(metrics, output_dir):
    """Plot training time comparison"""
    methods = []
    times = []
    
    for method, data in metrics.items():
        methods.append(method.upper())
        times.append(data.get("total_training_time", 0) / 3600)  # Convert to hours
    
    # Create figure
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(methods, times, color=colors[:len(methods)])
    
    # Customize
    plt.title("Training Time Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Training Time (hours)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}h',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'time_comparison.png')}")
    plt.close()


def plot_savings_breakdown(metrics, output_dir):
    """Plot comprehensive savings breakdown"""
    # Find baseline (assume first method or full finetuning)
    baseline_method = list(metrics.keys())[0]
    baseline_cost = metrics[baseline_method].get("total_cost_usd", 0)
    
    methods = []
    savings = []
    savings_pct = []
    
    for method, data in metrics.items():
        cost = data.get("total_cost_usd", 0)
        saving = baseline_cost - cost
        saving_pct = (saving / baseline_cost * 100) if baseline_cost > 0 else 0
        
        methods.append(method.upper())
        savings.append(saving)
        savings_pct.append(saving_pct)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute savings
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars1 = ax1.bar(methods, savings, color=colors[:len(methods)])
    ax1.set_title("Cost Savings (Absolute)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Savings (USD)", fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, save in zip(bars1, savings):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${save:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Percentage savings
    bars2 = ax2.bar(methods, savings_pct, color=colors[:len(methods)])
    ax2.set_title("Cost Savings (Percentage)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Savings (%)", fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, pct in zip(bars2, savings_pct):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle("Finetuning on a Budget: Savings Analysis", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "savings_breakdown.png"), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'savings_breakdown.png')}")
    plt.close()


def plot_comprehensive_comparison(metrics, output_dir):
    """Create a comprehensive multi-metric comparison"""
    # Prepare data
    methods = list(metrics.keys())
    
    # Normalize metrics to 0-1 scale for radar chart
    costs = [metrics[m].get("total_cost_usd", 0) for m in methods]
    times = [metrics[m].get("total_training_time", 0) / 3600 for m in methods]
    memory = [metrics[m].get("peak_memory_gb", 0) for m in methods]
    losses = [metrics[m].get("train_loss", 0) for m in methods]
    
    # Create comparison table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ["Method", "Cost (USD)", "Time (h)", "VRAM (GB)", "Final Loss"]
    
    for i, method in enumerate(methods):
        row = [
            method.upper(),
            f"${costs[i]:.2f}",
            f"{times[i]:.2f}",
            f"{memory[i]:.1f}",
            f"{losses[i]:.4f}" if losses[i] > 0 else "N/A"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = ['#FFE5E5', '#E5F9F9', '#E5F2F9']
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(colors[(i-1) % len(colors)])
    
    plt.title("Comprehensive Method Comparison", 
             fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, "comprehensive_comparison.png"), 
               dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'comprehensive_comparison.png')}")
    plt.close()


def main():
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metrics
    print("Loading training metrics...")
    metrics = load_metrics(args.results_dir)
    
    if not metrics:
        print("No metrics found. Please run training first.")
        return
    
    print(f"Found metrics for: {', '.join(metrics.keys())}")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_cost_comparison(metrics, args.output_dir)
    plot_memory_comparison(metrics, args.output_dir)
    plot_training_time_comparison(metrics, args.output_dir)
    plot_savings_breakdown(metrics, args.output_dir)
    plot_comprehensive_comparison(metrics, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"All plots saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

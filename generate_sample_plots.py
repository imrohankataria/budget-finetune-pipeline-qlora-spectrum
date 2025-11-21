#!/usr/bin/env python3
"""
Generate sample visualizations using the sample metrics
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory
os.makedirs("./results/plots", exist_ok=True)

# Load sample metrics
with open("./results/metrics/qlora_sample_metrics.json") as f:
    qlora_metrics = json.load(f)

with open("./results/metrics/spectrum_sample_metrics.json") as f:
    spectrum_metrics = json.load(f)

metrics = {
    "qlora": qlora_metrics,
    "spectrum": spectrum_metrics
}

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# 1. Cost Comparison
plt.figure(figsize=(10, 6))
methods = ["QLoRA", "Spectrum"]
costs = [qlora_metrics["total_cost_usd"], spectrum_metrics["total_cost_usd"]]
colors = ['#FF6B6B', '#4ECDC4']
bars = plt.bar(methods, costs, color=colors)

plt.title("Finetuning on a Budget: Cost Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Method", fontsize=12)
plt.ylabel("Total Cost (USD)", fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar, cost in zip(bars, costs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'${cost:.2f}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("./results/plots/cost_comparison.png", dpi=300, bbox_inches='tight')
print("Generated: cost_comparison.png")
plt.close()

# 2. Memory Comparison
plt.figure(figsize=(10, 6))
memory = [qlora_metrics["peak_memory_gb"], spectrum_metrics["peak_memory_gb"]]
bars = plt.bar(methods, memory, color=colors)

plt.title("VRAM Footprint Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Method", fontsize=12)
plt.ylabel("Peak VRAM (GB)", fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar, mem in zip(bars, memory):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{mem:.1f} GB',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("./results/plots/memory_comparison.png", dpi=300, bbox_inches='tight')
print("Generated: memory_comparison.png")
plt.close()

# 3. Training Time Comparison
plt.figure(figsize=(10, 6))
times = [qlora_metrics["total_training_time"] / 3600, 
         spectrum_metrics["total_training_time"] / 3600]
bars = plt.bar(methods, times, color=colors)

plt.title("Training Time Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Method", fontsize=12)
plt.ylabel("Training Time (hours)", fontsize=12)
plt.grid(axis='y', alpha=0.3)

for bar, time in zip(bars, times):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.2f}h',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("./results/plots/time_comparison.png", dpi=300, bbox_inches='tight')
print("Generated: time_comparison.png")
plt.close()

# 4. Savings Breakdown
baseline_cost = qlora_metrics["total_cost_usd"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

savings = [0, baseline_cost - spectrum_metrics["total_cost_usd"]]
savings_pct = [0, (savings[1] / baseline_cost * 100)]

bars1 = ax1.bar(methods, savings, color=colors)
ax1.set_title("Cost Savings (Absolute)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Savings (USD)", fontsize=12)
ax1.grid(axis='y', alpha=0.3)

for bar, save in zip(bars1, savings):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${save:.2f}',
            ha='center', va='bottom', fontweight='bold')

bars2 = ax2.bar(methods, savings_pct, color=colors)
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
plt.savefig("./results/plots/savings_breakdown.png", dpi=300, bbox_inches='tight')
print("Generated: savings_breakdown.png")
plt.close()

# 5. Comprehensive Comparison Table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ["QLoRA", f"${costs[0]:.2f}", f"{times[0]:.2f}", f"{memory[0]:.1f}", 
     f"{qlora_metrics['train_loss']:.4f}"],
    ["Spectrum", f"${costs[1]:.2f}", f"{times[1]:.2f}", f"{memory[1]:.1f}", 
     f"{spectrum_metrics['train_loss']:.4f}"]
]

headers = ["Method", "Cost (USD)", "Time (h)", "VRAM (GB)", "Final Loss"]

table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.5)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
colors_rows = ['#FFE5E5', '#E5F9F9']
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        table[(i, j)].set_facecolor(colors_rows[i-1])

plt.title("Comprehensive Method Comparison", 
         fontsize=16, fontweight='bold', pad=20)
plt.savefig("./results/plots/comprehensive_comparison.png", 
           dpi=300, bbox_inches='tight')
print("Generated: comprehensive_comparison.png")
plt.close()

# 6. Training Loss Curves
fig, ax = plt.subplots(figsize=(10, 6))

qlora_epochs = [m["epoch"] for m in qlora_metrics["epoch_metrics"]]
qlora_losses = [m["loss"] for m in qlora_metrics["epoch_metrics"]]
spectrum_epochs = [m["epoch"] for m in spectrum_metrics["epoch_metrics"]]
spectrum_losses = [m["loss"] for m in spectrum_metrics["epoch_metrics"]]

ax.plot(qlora_epochs, qlora_losses, 'o-', color='#FF6B6B', 
        linewidth=2, markersize=8, label='QLoRA')
ax.plot(spectrum_epochs, spectrum_losses, 's-', color='#4ECDC4', 
        linewidth=2, markersize=8, label='Spectrum')

ax.set_title("Training Loss Convergence", fontsize=16, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss", fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("./results/plots/loss_curves.png", dpi=300, bbox_inches='tight')
print("Generated: loss_curves.png")
plt.close()

print("\n" + "="*60)
print("All sample visualizations generated successfully!")
print("Location: ./results/plots/")
print("="*60)

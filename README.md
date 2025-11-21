# Budget Finetuning Pipeline: QLoRA vs Spectrum

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive benchmarking pipeline for low-cost finetuning techniques using LLaMA/Qwen models. This repository compares **QLoRA** (Quantized LoRA) and **Spectrum** (structured low-rank decomposition) approaches, measuring:

- 💰 **Training cost per epoch**
- 📈 **Accuracy curves and convergence speed**
- 🎮 **VRAM footprint**
- ⚡ **Training time and efficiency**

## 🌟 Features

- **Two State-of-the-Art Methods**: QLoRA and Spectrum implementations
- **Comprehensive Metrics**: Cost, memory, time, and accuracy tracking
- **Beautiful Visualizations**: "Finetuning on a Budget" comparison graphs
- **Evaluation Scripts**: Perplexity and generation quality assessment
- **Weight Artifacts**: Saved checkpoints and final models
- **Automated Benchmarking**: Single-command execution for full comparison

## 📊 Results Preview

The pipeline generates comparative visualizations including:
- Cost comparison across methods
- VRAM footprint analysis
- Training time benchmarks
- Savings breakdown (absolute & percentage)
- Comprehensive multi-metric tables

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/imrohankataria/budget-finetune-pipeline-qlora-spectrum.git
cd budget-finetune-pipeline-qlora-spectrum

# Install dependencies
pip install -r requirements.txt
```

### Run Full Benchmark

```bash
# Run both QLoRA and Spectrum training with evaluation
python run_benchmark.py --model_name meta-llama/Llama-2-7b-hf --num_epochs 3

# Run only specific methods
python run_benchmark.py --methods qlora
python run_benchmark.py --methods spectrum
```

## 📁 Project Structure

```
budget-finetune-pipeline-qlora-spectrum/
├── src/
│   ├── training/
│   │   ├── train_qlora.py         # QLoRA training script
│   │   └── train_spectrum.py      # Spectrum training script
│   ├── evaluation/
│   │   ├── evaluate_model.py      # Model evaluation script
│   │   └── visualize_results.py   # Visualization generation
│   └── utils/
│       ├── memory_tracker.py      # VRAM monitoring utilities
│       └── cost_calculator.py     # Cost calculation tools
├── configs/
│   ├── qlora_config.yaml          # QLoRA configuration
│   └── spectrum_config.yaml       # Spectrum configuration
├── results/
│   ├── qlora/                     # QLoRA outputs
│   ├── spectrum/                  # Spectrum outputs
│   ├── metrics/                   # Training metrics (JSON)
│   └── plots/                     # Comparison visualizations
├── run_benchmark.py               # Main benchmark runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Individual Training Scripts

### QLoRA Training

```bash
python src/training/train_qlora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset timdettmers/openassistant-guanaco \
    --output_dir ./results/qlora \
    --num_epochs 3 \
    --batch_size 4 \
    --lora_r 64 \
    --lora_alpha 16
```

**QLoRA Features:**
- 4-bit quantization with NF4
- Double quantization for extra memory savings
- LoRA adapters on attention layers
- Minimal trainable parameters (~0.1% of model)

### Spectrum Training

```bash
python src/training/train_spectrum.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset timdettmers/openassistant-guanaco \
    --output_dir ./results/spectrum \
    --num_epochs 3 \
    --batch_size 4 \
    --spectrum_rank 32 \
    --spectrum_alpha 16
```

**Spectrum Features:**
- Structured low-rank decomposition
- Spectral decomposition for efficiency
- Optional 8-bit quantization
- Balanced speed and quality

## 📊 Evaluation

### Evaluate a Trained Model

```bash
python src/evaluation/evaluate_model.py \
    --model_path ./results/qlora/final_model \
    --base_model meta-llama/Llama-2-7b-hf \
    --output_dir ./results/qlora/evaluation \
    --num_samples 500
```

**Metrics Calculated:**
- Perplexity on test set
- Average loss
- Generation quality samples
- Token-level accuracy

### Generate Visualizations

```bash
python src/evaluation/visualize_results.py \
    --results_dir ./results \
    --output_dir ./results/plots
```

**Generated Plots:**
- `cost_comparison.png` - Cost across methods
- `memory_comparison.png` - VRAM usage
- `time_comparison.png` - Training duration
- `savings_breakdown.png` - Cost savings analysis
- `comprehensive_comparison.png` - Multi-metric table

## 💡 Method Comparison

| Method | Quantization | Trainable Params | VRAM | Speed | Quality |
|--------|-------------|------------------|------|-------|---------|
| **QLoRA** | 4-bit (NF4) | ~0.1% | Low | Fast | High |
| **Spectrum** | 8-bit (opt) | ~0.2% | Medium | Very Fast | High |

### When to Use Each Method

**Use QLoRA when:**
- You need maximum memory efficiency
- Working with limited GPU memory (< 24GB)
- Training very large models (13B+)
- Quality is top priority

**Use Spectrum when:**
- You want the fastest training
- You have moderate GPU memory (24-40GB)
- You need quick iteration cycles
- Balance between speed and quality is important

## 🎯 Cost Analysis

The pipeline tracks and compares costs based on:
- GPU type and hourly rates (AWS/GCP pricing)
- Training time per epoch
- Memory efficiency
- Overall training duration

Example cost breakdown (A100 40GB):
- **Full Finetuning**: ~$4.00/hour
- **QLoRA**: ~$3.20/hour (20% savings)
- **Spectrum**: ~$3.00/hour (25% savings)

## 📈 Metrics Tracked

### Training Metrics
- Training loss per epoch
- Learning rate schedule
- Gradient norms
- Training time breakdown

### Memory Metrics
- Peak VRAM usage
- Average VRAM usage
- Memory allocation patterns
- System RAM usage

### Cost Metrics
- Cost per epoch (USD)
- Total training cost
- Cost savings vs baseline
- Cost efficiency ratio

### Quality Metrics
- Perplexity on evaluation set
- Generation quality samples
- Token prediction accuracy
- Convergence speed

## 🔬 Advanced Usage

### Custom Dataset

```python
# Modify training scripts to use custom data
python src/training/train_qlora.py \
    --dataset path/to/your/dataset \
    --max_seq_length 1024
```

### Different Models

```bash
# Use Qwen models
python run_benchmark.py --model_name Qwen/Qwen-7B

# Use different LLaMA variants
python run_benchmark.py --model_name meta-llama/Llama-2-13b-hf
```

### Custom Configurations

Edit `configs/qlora_config.yaml` or `configs/spectrum_config.yaml` to customize:
- Learning rates
- Rank parameters
- Batch sizes
- Quantization settings

## 📝 Output Files

### Training Outputs
- `results/{method}/final_model/` - Trained model weights
- `results/{method}/metrics/training_metrics.json` - Training statistics
- `results/{method}/logs/` - TensorBoard logs
- `results/{method}/checkpoints/` - Intermediate checkpoints

### Evaluation Outputs
- `results/{method}/evaluation/evaluation_metrics.json` - Evaluation results
- Sample generations and perplexity scores

### Visualization Outputs
- `results/plots/*.png` - All comparison graphs
- High-resolution (300 DPI) publication-ready figures

## 🛠️ Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size
--batch_size 2

# Reduce sequence length
--max_seq_length 256

# For QLoRA, increase gradient accumulation
# (edit training_args in script)
```

### Slow Training

```bash
# Reduce dataset size
# (edit prepare_dataset function to select fewer samples)

# Increase batch size if memory allows
--batch_size 8

# Use fewer epochs for quick tests
--num_epochs 1
```

## 📚 References

- **QLoRA**: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **LLaMA**: [Touvron et al., 2023](https://arxiv.org/abs/2302.13971)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional finetuning methods (DoRA, AdaLoRA, etc.)
- Support for more model architectures
- Enhanced evaluation metrics
- Multi-GPU training support

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for transformers and PEFT libraries
- Tim Dettmers for bitsandbytes and QLoRA
- Meta AI for LLaMA models

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Finetuning on a Budget! 💰🚀**
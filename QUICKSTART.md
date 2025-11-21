# Quick Start Example

This guide will help you run your first benchmark in under 5 minutes.

## Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 1-Minute Setup

```bash
# Clone and enter directory
git clone https://github.com/imrohankataria/budget-finetune-pipeline-qlora-spectrum.git
cd budget-finetune-pipeline-qlora-spectrum

# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (needed for LLaMA models)
huggingface-cli login
```

## Run Your First Benchmark

### Option 1: Quick Test (Recommended for First Run)

Test with minimal configuration:

```bash
python src/training/train_qlora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --num_epochs 1 \
    --output_dir ./results/qlora_test
```

This will:
- Download LLaMA 2 7B model (~13GB)
- Train for 1 epoch (~45 minutes on A100)
- Save results to `./results/qlora_test/`

### Option 2: Full Comparison Benchmark

Run both methods and compare:

```bash
python run_benchmark.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --num_epochs 1 \
    --methods all
```

This will:
- Train with both QLoRA and Spectrum
- Evaluate both models
- Generate comparison visualizations
- Take ~1.5-2 hours total

### Option 3: Generate Sample Plots (No Training)

See example visualizations without training:

```bash
python generate_sample_plots.py
```

View plots in `./results/plots/`

## Understanding the Output

After training completes, you'll find:

```
results/
├── qlora/
│   ├── final_model/           # Trained LoRA adapters
│   ├── metrics/
│   │   └── training_metrics.json  # Cost, time, memory stats
│   └── logs/                  # TensorBoard logs
├── spectrum/
│   └── (same structure)
└── plots/
    ├── cost_comparison.png
    ├── memory_comparison.png
    └── ... (more visualizations)
```

### View Training Metrics

```bash
# View metrics in terminal
cat results/qlora/metrics/training_metrics.json | python -m json.tool

# Key metrics to look for:
# - total_cost_usd: Total training cost
# - peak_memory_gb: Maximum VRAM used
# - train_loss: Final training loss
```

### View TensorBoard Logs

```bash
tensorboard --logdir ./results/qlora/logs
# Open http://localhost:6006 in browser
```

## Next Steps

1. **Compare Results**: Open the generated plots in `./results/plots/`
2. **Try Different Models**: Use `--model_name Qwen/Qwen-7B`
3. **Adjust Parameters**: Modify configs in `configs/` directory
4. **Read Full Docs**: Check `USAGE.md` for detailed documentation

## Common Issues

### Out of Memory
```bash
# Reduce batch size
python src/training/train_qlora.py --batch_size 2
```

### Slow Download
```bash
# Use a smaller model for testing
python run_benchmark.py --model_name gpt2
```

### Missing Dependencies
```bash
pip install --upgrade transformers peft bitsandbytes
```

## Sample Output

After successful training, you'll see:

```
============================================================
QLoRA Training Complete!
============================================================
Total training time: 2745.23 seconds
Final loss: 0.8245
Peak VRAM: 18.4 GB
Cost per epoch: $9.60
Total cost: $28.80
============================================================
```

## What's Next?

- 📖 Read the full [USAGE.md](USAGE.md) guide
- 🔧 Customize configs in `configs/`
- 📊 Explore evaluation scripts
- 🎨 Modify visualization styles
- 🚀 Train on your own dataset

## Getting Help

- Check [USAGE.md](USAGE.md) for detailed documentation
- Review [README.md](README.md) for project overview
- Open an issue on GitHub for bugs or questions

---

**Happy finetuning! 🎉**

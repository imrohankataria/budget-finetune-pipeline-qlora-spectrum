# USAGE.md - Detailed Usage Guide

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM)
- 50GB+ disk space for models and checkpoints

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/imrohankataria/budget-finetune-pipeline-qlora-spectrum.git
   cd budget-finetune-pipeline-qlora-spectrum
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Login to Hugging Face (for model downloads)**
   ```bash
   huggingface-cli login
   ```

## Running the Benchmark

### Full Benchmark Pipeline

Run both QLoRA and Spectrum training with complete evaluation:

```bash
python run_benchmark.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --num_epochs 3
```

This will:
1. Train model using QLoRA
2. Train model using Spectrum
3. Evaluate both models
4. Generate comparison visualizations

### Benchmark Specific Methods

**QLoRA Only:**
```bash
python run_benchmark.py --methods qlora --num_epochs 3
```

**Spectrum Only:**
```bash
python run_benchmark.py --methods spectrum --num_epochs 3
```

**Skip Evaluation:**
```bash
python run_benchmark.py --skip_evaluation
```

**Skip Visualization:**
```bash
python run_benchmark.py --skip_visualization
```

## Individual Training Scripts

### QLoRA Training

**Basic Usage:**
```bash
python src/training/train_qlora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ./results/qlora \
    --num_epochs 3
```

**Advanced Options:**
```bash
python src/training/train_qlora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset timdettmers/openassistant-guanaco \
    --output_dir ./results/qlora \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --max_seq_length 512 \
    --logging_steps 10 \
    --save_steps 100
```

**Available Parameters:**
- `--model_name`: Base model (default: meta-llama/Llama-2-7b-hf)
- `--dataset`: Training dataset (default: timdettmers/openassistant-guanaco)
- `--output_dir`: Output directory (default: ./results/qlora)
- `--num_epochs`: Number of epochs (default: 3)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.1)
- `--max_seq_length`: Max sequence length (default: 512)

### Spectrum Training

**Basic Usage:**
```bash
python src/training/train_spectrum.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --output_dir ./results/spectrum \
    --num_epochs 3
```

**Advanced Options:**
```bash
python src/training/train_spectrum.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset timdettmers/openassistant-guanaco \
    --output_dir ./results/spectrum \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --spectrum_rank 32 \
    --spectrum_alpha 16 \
    --max_seq_length 512 \
    --use_8bit
```

**Available Parameters:**
- `--model_name`: Base model
- `--dataset`: Training dataset
- `--output_dir`: Output directory (default: ./results/spectrum)
- `--num_epochs`: Number of epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--spectrum_rank`: Spectrum rank (default: 32)
- `--spectrum_alpha`: Spectrum alpha (default: 16)
- `--use_8bit`: Enable 8-bit quantization (flag)

## Evaluation

### Evaluate a Trained Model

```bash
python src/evaluation/evaluate_model.py \
    --model_path ./results/qlora/final_model \
    --base_model meta-llama/Llama-2-7b-hf \
    --output_dir ./results/qlora/evaluation \
    --num_samples 500 \
    --batch_size 4
```

**Output:**
- Perplexity score
- Average loss
- Sample generations
- Metrics saved to JSON

### Generate Visualizations

```bash
python src/evaluation/visualize_results.py \
    --results_dir ./results \
    --output_dir ./results/plots
```

**Generated Plots:**
1. `cost_comparison.png` - Cost across methods
2. `memory_comparison.png` - VRAM usage comparison
3. `time_comparison.png` - Training time comparison
4. `savings_breakdown.png` - Absolute and percentage savings
5. `comprehensive_comparison.png` - Multi-metric comparison table
6. `loss_curves.png` - Training loss convergence

## Working with Different Models

### LLaMA Models

```bash
# LLaMA 2 7B
python run_benchmark.py --model_name meta-llama/Llama-2-7b-hf

# LLaMA 2 13B (requires more VRAM)
python run_benchmark.py --model_name meta-llama/Llama-2-13b-hf
```

### Qwen Models

```bash
# Qwen 7B
python run_benchmark.py --model_name Qwen/Qwen-7B

# Qwen 14B
python run_benchmark.py --model_name Qwen/Qwen-14B
```

### Other Compatible Models

Any Hugging Face model compatible with `AutoModelForCausalLM` can be used.

## Working with Custom Datasets

### From Hugging Face Hub

```bash
python src/training/train_qlora.py \
    --dataset your-username/your-dataset \
    --model_name meta-llama/Llama-2-7b-hf
```

### From Local Files

Modify the `prepare_dataset` function in the training scripts to load from local files:

```python
# In train_qlora.py or train_spectrum.py
from datasets import load_dataset

# Load from JSON
dataset = load_dataset('json', data_files='data/train.json', split='train')

# Load from CSV
dataset = load_dataset('csv', data_files='data/train.csv', split='train')
```

## Configuration Files

### Using YAML Configs

Edit `configs/qlora_config.yaml` or `configs/spectrum_config.yaml`:

```yaml
model_name: "meta-llama/Llama-2-7b-hf"
dataset: "timdettmers/openassistant-guanaco"
num_epochs: 3
batch_size: 4
learning_rate: 0.0002
```

## Monitoring Training

### TensorBoard

Training logs are automatically saved to TensorBoard format:

```bash
tensorboard --logdir ./results/qlora/logs
tensorboard --logdir ./results/spectrum/logs
```

Access at http://localhost:6006

### Weights & Biases (WandB)

To enable WandB logging, modify training scripts:

```python
training_args = TrainingArguments(
    ...
    report_to=["tensorboard", "wandb"],
)
```

Then run with WandB authentication:
```bash
wandb login
python src/training/train_qlora.py ...
```

## Memory Optimization Tips

### Out of Memory Issues

1. **Reduce batch size:**
   ```bash
   --batch_size 2
   ```

2. **Reduce sequence length:**
   ```bash
   --max_seq_length 256
   ```

3. **Increase gradient accumulation** (edit training_args in script):
   ```python
   gradient_accumulation_steps=8
   ```

4. **Use smaller LoRA rank:**
   ```bash
   --lora_r 32
   ```

### Maximizing Throughput

1. **Increase batch size** (if memory allows):
   ```bash
   --batch_size 8
   ```

2. **Optimize data loading:**
   ```python
   dataloader_num_workers=8
   ```

3. **Use gradient checkpointing** (add to training script):
   ```python
   model.gradient_checkpointing_enable()
   ```

## Expected Results

### Typical Metrics (LLaMA 2 7B, 3 epochs)

**QLoRA:**
- Training time: ~2.5 hours
- Peak VRAM: 18-20 GB
- Cost: ~$29 (A100)
- Final loss: ~0.8
- Trainable params: ~4M (0.06%)

**Spectrum:**
- Training time: ~2.2 hours
- Peak VRAM: 22-24 GB
- Cost: ~$26 (A100)
- Final loss: ~0.85
- Trainable params: ~13M (0.2%)

## Troubleshooting

### CUDA Out of Memory

Reduce memory usage with the tips above, or switch to a larger GPU.

### Slow Training

- Ensure CUDA is properly installed: `torch.cuda.is_available()`
- Check GPU utilization: `nvidia-smi`
- Increase `dataloader_num_workers`

### Model Download Issues

- Check Hugging Face authentication: `huggingface-cli whoami`
- For gated models (LLaMA), request access on Hugging Face Hub
- Use `HF_HUB_OFFLINE=1` for offline mode

### Import Errors

```bash
pip install --upgrade transformers peft bitsandbytes accelerate
```

## Advanced Usage

### Multi-GPU Training

The scripts support multi-GPU training via `device_map="auto"`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/training/train_qlora.py ...
```

### Custom LoRA Target Modules

Edit the `lora_config` in training scripts:

```python
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ...
)
```

### Resume from Checkpoint

```bash
python src/training/train_qlora.py \
    --output_dir ./results/qlora \
    --resume_from_checkpoint ./results/qlora/checkpoint-100
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{budget-finetune-pipeline,
  author = {Rohan Kataria},
  title = {Budget Finetuning Pipeline: QLoRA vs Spectrum},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/imrohankataria/budget-finetune-pipeline-qlora-spectrum}
}
```

"""
Spectrum Finetuning Script
Implements parameter-efficient finetuning using Spectrum technique
Spectrum uses structured low-rank decomposition for efficient finetuning
"""

import os
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from utils.memory_tracker import MemoryTracker
from utils.cost_calculator import CostCalculator


class SpectrumAdapter(nn.Module):
    """
    Spectrum adapter for efficient parameter finetuning
    Uses spectral decomposition with low-rank structure
    """
    def __init__(self, in_features, out_features, rank=32, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Spectral decomposition matrices
        self.spectrum_down = nn.Linear(in_features, rank, bias=False)
        self.spectrum_up = nn.Linear(rank, out_features, bias=False)
        
        # Initialize with small random values
        nn.init.kaiming_uniform_(self.spectrum_down.weight, a=1)
        nn.init.zeros_(self.spectrum_up.weight)
        
    def forward(self, x):
        # Apply spectrum decomposition
        return self.spectrum_up(self.spectrum_down(x)) * self.scaling


def add_spectrum_adapters(model, rank=32, alpha=16):
    """Add Spectrum adapters to the model"""
    
    adapter_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in 
                                                   ["q_proj", "k_proj", "v_proj", "o_proj"]):
            # Get dimensions
            in_features = module.in_features
            out_features = module.out_features
            
            # Create spectrum adapter
            adapter = SpectrumAdapter(in_features, out_features, rank, alpha)
            
            # Register as a submodule
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, f"{name.split('.')[-1]}_spectrum", adapter)
                adapter_modules.append((name, adapter))
    
    # Freeze original parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only spectrum adapters
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'spectrum' in name:
            param.requires_grad = True
            trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.2f}%")
    
    return model, adapter_modules


class SpectrumModel(nn.Module):
    """Wrapper for model with Spectrum adapters"""
    
    def __init__(self, base_model, adapters):
        super().__init__()
        self.base_model = base_model
        self.adapters = nn.ModuleDict({name: adapter for name, adapter in adapters})
        
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Spectrum Finetuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model name")
    parser.add_argument("--dataset", type=str, default="timdettmers/openassistant-guanaco",
                        help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="./results/spectrum",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--spectrum_rank", type=int, default=32,
                        help="Spectrum rank")
    parser.add_argument("--spectrum_alpha", type=int, default=16,
                        help="Spectrum alpha")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint interval")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    return parser.parse_args()


def load_and_prepare_model(model_name, spectrum_rank, spectrum_alpha, use_8bit=False):
    """Load model and add Spectrum adapters"""
    
    # Load model
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    
    if use_8bit:
        load_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs
    )
    
    # Add Spectrum adapters
    model, adapters = add_spectrum_adapters(model, spectrum_rank, spectrum_alpha)
    
    return model


def prepare_dataset(dataset_name, tokenizer, max_seq_length):
    """Load and prepare dataset for training"""
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Take a subset for faster benchmarking
    dataset = dataset.select(range(min(5000, len(dataset))))
    
    def tokenize_function(examples):
        # Tokenize the text
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset


def main():
    args = parse_args()
    
    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize trackers
    memory_tracker = MemoryTracker()
    cost_calculator = CostCalculator()
    
    print(f"Starting Spectrum finetuning of {args.model_name}")
    print(f"Configuration: {vars(args)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare model
    print("\nLoading model with Spectrum adapters...")
    start_time = time.time()
    model = load_and_prepare_model(
        args.model_name,
        args.spectrum_rank,
        args.spectrum_alpha,
        args.use_8bit
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Track initial memory
    memory_stats = memory_tracker.get_memory_stats()
    print(f"\nInitial VRAM usage: {memory_stats['allocated_gb']:.2f} GB")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(args.dataset, tokenizer, args.max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        report_to=["tensorboard"],
        warmup_steps=100,
        weight_decay=0.01,
        dataloader_num_workers=4,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Track training metrics
    training_metrics = {
        "method": "spectrum",
        "model_name": args.model_name,
        "start_time": datetime.now().isoformat(),
        "config": vars(args),
        "memory_stats": [],
        "epoch_metrics": [],
    }
    
    # Train
    print("\nStarting training...")
    train_start = time.time()
    
    result = trainer.train()
    
    train_time = time.time() - train_start
    
    # Final memory stats
    final_memory = memory_tracker.get_memory_stats()
    peak_memory = memory_tracker.get_peak_memory()
    
    # Calculate costs
    cost_per_epoch = cost_calculator.calculate_training_cost(
        train_time / args.num_epochs,
        peak_memory,
        "spectrum"
    )
    
    # Save metrics
    training_metrics.update({
        "end_time": datetime.now().isoformat(),
        "total_training_time": train_time,
        "train_loss": result.training_loss,
        "peak_memory_gb": peak_memory,
        "final_memory_gb": final_memory['allocated_gb'],
        "cost_per_epoch_usd": cost_per_epoch,
        "total_cost_usd": cost_per_epoch * args.num_epochs,
    })
    
    # Save results
    metrics_file = os.path.join(metrics_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Spectrum Training Complete!")
    print(f"{'='*60}")
    print(f"Total training time: {train_time:.2f} seconds")
    print(f"Final loss: {result.training_loss:.4f}")
    print(f"Peak VRAM: {peak_memory:.2f} GB")
    print(f"Cost per epoch: ${cost_per_epoch:.4f}")
    print(f"Total cost: ${cost_per_epoch * args.num_epochs:.4f}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"{'='*60}")
    
    # Save model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    print(f"\nModel saved to: {os.path.join(args.output_dir, 'final_model')}")


if __name__ == "__main__":
    main()

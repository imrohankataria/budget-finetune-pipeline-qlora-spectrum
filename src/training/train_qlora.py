"""
QLoRA Finetuning Script
Implements parameter-efficient finetuning using QLoRA (Quantized LoRA)
"""

import os
import time
import json
import argparse
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

from utils.memory_tracker import MemoryTracker
from utils.cost_calculator import CostCalculator


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Finetuning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model name")
    parser.add_argument("--dataset", type=str, default="timdettmers/openassistant-guanaco",
                        help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="./results/qlora",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint interval")
    return parser.parse_args()


def load_and_prepare_model(model_name, lora_config):
    """Load model with 4-bit quantization and prepare for QLoRA training"""
    
    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
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
    
    print(f"Starting QLoRA finetuning of {args.model_name}")
    print(f"Configuration: {vars(args)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load and prepare model
    print("\nLoading model with 4-bit quantization...")
    start_time = time.time()
    model = load_and_prepare_model(args.model_name, lora_config)
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
        "method": "qlora",
        "model_name": args.model_name,
        "start_time": datetime.now().isoformat(),
        "config": vars(args),
        "memory_stats": [],
        "epoch_metrics": [],
    }
    
    # Train
    print("\nStarting training...")
    train_start = time.time()
    
    # Training loop with memory tracking
    result = trainer.train()
    
    train_time = time.time() - train_start
    
    # Final memory stats
    final_memory = memory_tracker.get_memory_stats()
    peak_memory = memory_tracker.get_peak_memory()
    
    # Calculate costs
    cost_per_epoch = cost_calculator.calculate_training_cost(
        train_time / args.num_epochs,
        peak_memory,
        "qlora"
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
    print("QLoRA Training Complete!")
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

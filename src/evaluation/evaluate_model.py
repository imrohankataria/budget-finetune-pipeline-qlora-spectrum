"""
Model evaluation script for measuring accuracy and performance
"""

import os
import json
import argparse
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuned model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model name")
    parser.add_argument("--dataset", type=str, default="timdettmers/openassistant-guanaco",
                        help="Evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="./results/evaluation",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    
    return model, tokenizer


def calculate_perplexity(model, tokenizer, texts, batch_size=4, max_length=512):
    """Calculate perplexity on a set of texts"""
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(model.device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Accumulate loss
            loss = outputs.loss
            num_tokens = inputs["attention_mask"].sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def evaluate_generation_quality(model, tokenizer, prompts, max_new_tokens=50):
    """Evaluate generation quality on prompts"""
    generations = []
    
    model.eval()
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating responses"):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append({
                "prompt": prompt,
                "generation": generated_text,
            })
    
    return generations


def main():
    args = parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    # Load evaluation dataset
    print(f"\nLoading evaluation dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    eval_samples = dataset.select(range(min(args.num_samples, len(dataset))))
    
    # Extract texts
    texts = [sample["text"] for sample in eval_samples]
    
    # Calculate perplexity
    print("\nCalculating perplexity...")
    perplexity, avg_loss = calculate_perplexity(
        model, tokenizer, texts,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Evaluate generation quality on subset
    print("\nEvaluating generation quality...")
    test_prompts = texts[:10]  # Use first 10 as test prompts
    generations = evaluate_generation_quality(model, tokenizer, test_prompts)
    
    # Collect metrics
    metrics = {
        "model_path": args.model_path,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "perplexity": float(perplexity),
        "avg_loss": float(avg_loss),
        "evaluation_time": datetime.now().isoformat(),
        "sample_generations": generations[:5],  # Save first 5 examples
    }
    
    # Save metrics
    output_file = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

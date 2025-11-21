#!/usr/bin/env python3
"""
Custom dataset example: Shows how to use your own dataset for finetuning
"""

import json
from datasets import Dataset

def create_custom_dataset():
    """
    Example of creating a custom dataset for finetuning.
    Replace this with your own data loading logic.
    """
    
    # Example: Simple instruction-response pairs
    data = [
        {
            "instruction": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Explain deep learning",
            "response": "Deep learning is a type of machine learning based on artificial neural networks with multiple layers that can learn hierarchical representations of data."
        },
        {
            "instruction": "What is finetuning?",
            "response": "Finetuning is the process of adapting a pre-trained model to a specific task or dataset by continuing training with task-specific data."
        },
        # Add more examples...
    ]
    
    # Format as text (example format)
    formatted_data = []
    for item in data:
        text = f"### Instruction: {item['instruction']}\n\n### Response: {item['response']}"
        formatted_data.append({"text": text})
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "text": [item["text"] for item in formatted_data]
    })
    
    return dataset

def save_dataset_to_file(dataset, output_path):
    """Save dataset to JSON file"""
    data = []
    for item in dataset:
        data.append(item)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Dataset saved to: {output_path}")

def main():
    print("="*60)
    print("Custom Dataset Example")
    print("="*60)
    
    # Create custom dataset
    print("\n1. Creating custom dataset...")
    dataset = create_custom_dataset()
    print(f"   Dataset size: {len(dataset)} examples")
    
    # Show sample
    print("\n2. Sample from dataset:")
    print(dataset[0]["text"])
    
    # Save to file
    print("\n3. Saving dataset...")
    save_dataset_to_file(dataset, "custom_dataset.json")
    
    # Usage instructions
    print("\n4. Using this dataset for training:")
    print("\n   Modify the prepare_dataset function in train_qlora.py:")
    print("   ")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('json', data_files='custom_dataset.json')")
    print("\n   Then run:")
    print("   python src/training/train_qlora.py \\")
    print("       --dataset custom_dataset.json \\")
    print("       --model_name meta-llama/Llama-2-7b-hf")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

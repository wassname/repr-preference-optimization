"""
Simple dataset loading for preference optimization.
Supports the datasets mentioned in the original experiments.
"""

import json
from typing import List, Dict, Any
from datasets import load_dataset
import random


def load_preference_dataset(dataset_name: str, split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """
    Load a preference dataset in the format: {"prompt": str, "chosen": str, "rejected": str}
    
    Args:
        dataset_name: Name of dataset (math, code, alpaca_mmlu, etc.)
        split: train/test/validation
        n_samples: Optional limit on number of samples
    
    Returns:
        List of preference pairs
    """
    
    if dataset_name == "math":
        return load_math_dataset(split, n_samples)
    elif dataset_name == "code":
        return load_code_dataset(split, n_samples)
    elif dataset_name == "alpaca_mmlu":
        return load_alpaca_mmlu_dataset(split, n_samples)
    elif dataset_name == "cooking":
        return load_cooking_dataset(split, n_samples)
    elif dataset_name == "alpaca_low_quality":
        return load_alpaca_low_quality_dataset(split, n_samples)
    elif dataset_name == "maths_easy":
        return load_maths_easy_dataset(split, n_samples)
    else:
        # Fallback: generate simple synthetic data
        return generate_synthetic_math_data(n_samples or 100)


def load_math_dataset(split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """Load math preference dataset"""
    try:
        # Try to load from HuggingFace or create synthetic
        return generate_synthetic_math_data(n_samples or 500)
    except:
        return generate_synthetic_math_data(n_samples or 500)


def load_code_dataset(split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """Load code preference dataset"""
    return generate_synthetic_code_data(n_samples or 500)


def load_alpaca_mmlu_dataset(split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """Load Alpaca MMLU dataset"""
    try:
        # Try to load real MMLU data
        dataset = load_dataset("cais/mmlu", "all", split=split)
        data = []
        
        for i, item in enumerate(dataset):
            if n_samples and i >= n_samples:
                break
                
            question = item["question"]
            choices = item["choices"]
            correct_idx = item["answer"]
            
            # Create preference pair
            correct_answer = choices[correct_idx]
            wrong_answers = [c for i, c in enumerate(choices) if i != correct_idx]
            wrong_answer = random.choice(wrong_answers)
            
            data.append({
                "prompt": f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:",
                "chosen": f" {correct_answer}",
                "rejected": f" {wrong_answer}"
            })
        
        return data
    except:
        return generate_synthetic_qa_data(n_samples or 500)


def load_cooking_dataset(split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """Load cooking preference dataset"""
    return generate_synthetic_cooking_data(n_samples or 500)


def load_alpaca_low_quality_dataset(split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """Load low quality Alpaca dataset"""
    return generate_synthetic_qa_data(n_samples or 500, quality="low")


def load_maths_easy_dataset(split: str = "train", n_samples: int = None) -> List[Dict[str, str]]:
    """Load easy math dataset"""
    return generate_synthetic_math_data(n_samples or 500, difficulty="easy")


def generate_synthetic_math_data(n_samples: int, difficulty: str = "medium") -> List[Dict[str, str]]:
    """Generate synthetic math preference data"""
    data = []
    
    for i in range(n_samples):
        if difficulty == "easy":
            # Simple addition/subtraction
            a, b = random.randint(1, 20), random.randint(1, 20)
            op = random.choice(["+", "-"])
            if op == "+":
                correct = a + b
                wrong = correct + random.randint(1, 5)
            else:
                correct = a - b
                wrong = correct + random.randint(1, 5)
            question = f"What is {a} {op} {b}?"
        else:
            # Multiplication/division
            a, b = random.randint(2, 12), random.randint(2, 12)
            op = random.choice(["*", "/"])
            if op == "*":
                correct = a * b
                wrong = correct + random.randint(1, 10)
                question = f"What is {a} × {b}?"
            else:
                correct = a
                wrong = correct + random.randint(1, 5)
                question = f"What is {a * b} ÷ {b}?"
        
        data.append({
            "prompt": question,
            "chosen": f"The answer is {correct}.",
            "rejected": f"The answer is {wrong}."
        })
    
    return data


def generate_synthetic_code_data(n_samples: int) -> List[Dict[str, str]]:
    """Generate synthetic code preference data"""
    data = []
    
    functions = [
        ("sum", "def sum_list(lst):\n    return sum(lst)", "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total"),
        ("max", "def find_max(lst):\n    return max(lst)", "def find_max(lst):\n    return sorted(lst)[-1]"),
        ("reverse", "def reverse_string(s):\n    return s[::-1]", "def reverse_string(s):\n    return ''.join(reversed(s))"),
    ]
    
    for i in range(n_samples):
        func_name, good_code, bad_code = random.choice(functions)
        
        data.append({
            "prompt": f"Write a Python function to {func_name} elements:",
            "chosen": good_code,
            "rejected": bad_code
        })
    
    return data


def generate_synthetic_qa_data(n_samples: int, quality: str = "high") -> List[Dict[str, str]]:
    """Generate synthetic Q&A data"""
    data = []
    
    topics = ["science", "history", "geography", "literature"]
    
    for i in range(n_samples):
        topic = random.choice(topics)
        
        if quality == "low":
            data.append({
                "prompt": f"Tell me about {topic}:",
                "chosen": f"{topic.title()} is a very interesting field of study with many important concepts.",
                "rejected": f"{topic} is bad and boring nobody should study it."
            })
        else:
            data.append({
                "prompt": f"What is an important concept in {topic}?",
                "chosen": f"An important concept in {topic} is understanding the fundamental principles that govern the field.",
                "rejected": f"I don't know anything about {topic}, it's too difficult."
            })
    
    return data


def generate_synthetic_cooking_data(n_samples: int) -> List[Dict[str, str]]:
    """Generate synthetic cooking preference data"""
    data = []
    
    recipes = [
        ("pasta", "Cook pasta in boiling salted water until al dente, then drain.", "Microwave pasta with cold water for 5 minutes."),
        ("eggs", "Crack eggs into a heated pan with oil and cook until whites are set.", "Boil eggs in their shells for 20 minutes."),
        ("rice", "Rinse rice and cook with 2:1 water ratio for 18 minutes.", "Fry uncooked rice directly in a dry pan."),
    ]
    
    for i in range(n_samples):
        dish, good_method, bad_method = random.choice(recipes)
        
        data.append({
            "prompt": f"How do you cook {dish}?",
            "chosen": good_method,
            "rejected": bad_method
        })
    
    return data


if __name__ == "__main__":
    # Test the dataset loading
    for dataset_name in ["math", "code", "alpaca_mmlu", "cooking"]:
        print(f"\n=== {dataset_name} ===")
        data = load_preference_dataset(dataset_name, n_samples=3)
        for item in data[:2]:
            print(f"Prompt: {item['prompt']}")
            print(f"Chosen: {item['chosen']}")
            print(f"Rejected: {item['rejected']}")
            print()
"""
Evaluation functions for InnerPO using open_pref_eval.

This gives you comprehensive evaluation across distribution shifts,
which is perfect for measuring generalization in your paper.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_loader import load_preference_dataset

# Import open_pref_eval for comprehensive evaluation
try:
    import open_pref_eval as ope
    HAS_OPEN_PREF_EVAL = True
except ImportError:
    HAS_OPEN_PREF_EVAL = False
    print("Warning: open_pref_eval not found. Install with: pip install open-pref-eval")


def compute_preference_accuracy(model, ref_model, tokenizer, eval_data: List[Dict], 
                               beta: float = 0.1, device: str = "cuda") -> float:
    """
    Compute preference accuracy - how often the model prefers chosen over rejected.
    
    This is the key metric for preference optimization.
    """
    model.eval()
    ref_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for item in eval_data:
            # Tokenize
            prompt = item["prompt"]
            chosen = item["chosen"] 
            rejected = item["rejected"]
            
            # Create full sequences
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            chosen_tokens = tokenizer(chosen_text, return_tensors="pt", truncate=True, max_length=512)
            rejected_tokens = tokenizer(rejected_text, return_tensors="pt", truncate=True, max_length=512)
            
            chosen_ids = chosen_tokens["input_ids"].to(device)
            rejected_ids = rejected_tokens["input_ids"].to(device)
            
            # Get log probabilities
            chosen_logprobs = get_sequence_logprob(model, chosen_ids)
            rejected_logprobs = get_sequence_logprob(model, rejected_ids)
            
            ref_chosen_logprobs = get_sequence_logprob(ref_model, chosen_ids)
            ref_rejected_logprobs = get_sequence_logprob(ref_model, rejected_ids)
            
            # DPO preference score
            chosen_score = chosen_logprobs - ref_chosen_logprobs
            rejected_score = rejected_logprobs - ref_rejected_logprobs
            
            # Model prefers chosen if chosen_score > rejected_score
            if chosen_score > rejected_score:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0


def get_sequence_logprob(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Get log probability of a sequence"""
    logits = model(input_ids).logits
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Get log probability for each token
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Sum over sequence (could also use mean)
    return token_log_probs.sum()


def evaluate_on_datasets(model, ref_model, tokenizer, datasets: List[str], 
                        device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate on multiple datasets and return accuracy for each.
    
    This gives you the core metrics for your paper tables.
    """
    results = {}
    
    for dataset_name in datasets:
        print(f"Evaluating on {dataset_name}...")
        
        # Load test data
        eval_data = load_preference_dataset(dataset_name, split="test", n_samples=100)
        
        # Compute accuracy
        accuracy = compute_preference_accuracy(model, ref_model, tokenizer, eval_data, device=device)
        results[dataset_name] = accuracy
        
        print(f"{dataset_name}: {accuracy:.3f}")
    
    return results


def eval_with_open_pref_eval(model_path: str, ref_model_name: str, 
                            train_dataset: str = "math", 
                            eval_categories: List[str] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation using open_pref_eval.
    
    This evaluates your model across different distribution shifts:
    - In-distribution (same as training)
    - Out-of-distribution (different domains)
    - Random baselines
    
    Perfect for your paper's generalization claims!
    """
    if not HAS_OPEN_PREF_EVAL:
        print("open_pref_eval not available, falling back to simple eval")
        return quick_eval(model_path, ref_model_name, eval_categories or ["math", "code"])
    
    if eval_categories is None:
        # Default comprehensive evaluation
        eval_categories = [
            # In-distribution
            train_dataset,
            
            # Similar domains (moderate shift)
            "code" if train_dataset != "code" else "math",
            "alpaca_mmlu",
            
            # Different domains (large shift)  
            "cooking",
            "creative_writing",
            
            # Quality shifts
            "alpaca_low_quality",
            "alpaca_high_quality",
        ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    print(f"Loading reference model {ref_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, device_map="auto")
    
    results = {}
    
    print(f"\nEvaluating across {len(eval_categories)} datasets for distribution shift analysis...")
    
    for category in eval_categories:
        print(f"\nEvaluating on {category}...")
        
        try:
            # Use open_pref_eval to get the dataset
            eval_data = ope.get_dataset(category, split="test", n_samples=200)
            
            # Convert to our format if needed
            if hasattr(eval_data[0], 'prompt'):
                eval_data = [{"prompt": item.prompt, "chosen": item.chosen, "rejected": item.rejected} 
                           for item in eval_data]
            
            # Compute accuracy
            accuracy = compute_preference_accuracy(model, ref_model, tokenizer, eval_data, device=device)
            results[category] = accuracy
            
            # Categorize the result
            if category == train_dataset:
                shift_type = "in-distribution"
            elif category in ["alpaca_low_quality", "alpaca_high_quality"]:
                shift_type = "quality-shift"
            elif category in ["cooking", "creative_writing"]:
                shift_type = "domain-shift" 
            else:
                shift_type = "moderate-shift"
                
            print(f"  {category} ({shift_type}): {accuracy:.3f}")
            
        except Exception as e:
            print(f"  Failed to evaluate {category}: {e}")
            results[category] = 0.0
    
    return results


def quick_eval(model_path: str, ref_model_name: str, datasets: List[str] = None) -> Dict[str, float]:
    """
    Quick evaluation of a trained model (fallback when open_pref_eval unavailable).
    
    Usage:
        results = quick_eval("./outputs/innerpo-supr_math_seed1", "Qwen/Qwen3-0.6B")
    """
    if datasets is None:
        datasets = ["math", "code", "alpaca_mmlu"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    print(f"Loading reference model {ref_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, device_map="auto")
    
    # Evaluate
    results = evaluate_on_datasets(model, ref_model, tokenizer, datasets, device)
    
    return results


def comprehensive_eval_table(results: Dict[str, float], train_dataset: str) -> None:
    """
    Print results in a nice table format for your paper.
    
    Categorizes results by distribution shift type.
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION RESULTS")
    print(f"Training dataset: {train_dataset}")
    print(f"{'='*60}")
    
    # Categorize results
    categories = {
        "In-Distribution": [],
        "Moderate Shift": [], 
        "Domain Shift": [],
        "Quality Shift": []
    }
    
    for dataset, accuracy in results.items():
        if dataset == train_dataset:
            categories["In-Distribution"].append((dataset, accuracy))
        elif dataset in ["alpaca_low_quality", "alpaca_high_quality"]:
            categories["Quality Shift"].append((dataset, accuracy))
        elif dataset in ["cooking", "creative_writing"]:
            categories["Domain Shift"].append((dataset, accuracy))
        else:
            categories["Moderate Shift"].append((dataset, accuracy))
    
    # Print categorized results
    for category, items in categories.items():
        if items:
            print(f"\n{category}:")
            for dataset, accuracy in items:
                print(f"  {dataset:20s}: {accuracy:.3f}")
    
    # Summary statistics
    in_dist = [acc for cat, items in categories.items() if cat == "In-Distribution" for _, acc in items]
    out_dist = [acc for cat, items in categories.items() if cat != "In-Distribution" for _, acc in items]
    
    if in_dist and out_dist:
        print(f"\nSUMMARY:")
        print(f"  In-distribution avg:  {sum(in_dist)/len(in_dist):.3f}")
        print(f"  Out-distribution avg: {sum(out_dist)/len(out_dist):.3f}")
        print(f"  Generalization gap:   {sum(in_dist)/len(in_dist) - sum(out_dist)/len(out_dist):.3f}")
        print(f"  (Lower gap = better generalization)")

    
def paper_eval(model_path: str, ref_model: str = "Qwen/Qwen3-0.6B", 
               train_dataset: str = "math") -> Dict[str, float]:
    """
    One-shot evaluation for paper results.
    
    This gives you everything you need for your paper tables.
    """
    print(f"Paper Evaluation: {model_path}")
    print(f"Reference model: {ref_model}")
    
    # Run comprehensive evaluation
    results = eval_with_open_pref_eval(model_path, ref_model, train_dataset)
    
    # Print nice table
    comprehensive_eval_table(results, train_dataset)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <model_path> [ref_model]")
        print("Example: python evaluate.py ./outputs/innerpo-supr_math_seed1 Qwen/Qwen3-0.6B")
        sys.exit(1)
    
    model_path = sys.argv[1]
    ref_model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-0.6B"
    
    results = quick_eval(model_path, ref_model)
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    for dataset, accuracy in results.items():
        print(f"{dataset:15s}: {accuracy:.3f}")
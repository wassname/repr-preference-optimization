# Simplified InnerPO Justfile
export CUDA_VISIBLE_DEVICES := "0"
export TOKENIZERS_PARALLELISM := "false"

[private]
default:
    @just --list

# Generate and run the full experimental sweep
sweep:
    #!/usr/bin/bash
    mv outputs outputs_$(date +%Y-%m-%d_%H-%M-%S) || true
    python sweep.py > sweep.sh
    chmod +x sweep.sh
    ./sweep.sh 2>&1 | tee sweep.txt

# Run a single experiment with hierarchical CLI
run method *args:
    python cli.py train {{method}} {{args}}

# Run with dev config (small model, few samples)
dev method *args:
    python cli.py train {{method}} --model.base_model=reciprocate/tiny-llama --training.max_steps=50 --system.verbose=2 {{args}}

# Test a single method quickly
test method:
    python cli.py train {{method}} --model.base_model=reciprocate/tiny-llama --training.max_steps=10 --data.batch_size=2 --system.verbose=2

# Evaluate a trained model with distribution shift analysis
eval model_path *args:
    python cli.py eval {{model_path}} {{args}}

# Show CLI examples
examples:
    python cli.py config

# Install dependencies
install:
    uv sync

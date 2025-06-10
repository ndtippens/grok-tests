#!/bin/bash

# Test all experiment configurations with a small number of steps
set -e

echo "Testing all experiment configurations..."

experiments=(
    "baseline"
    "l1_only" 
    "grokfast_only"
    "grokfast_ma_only"
    "l1_plus_grokfast"
    "l1_plus_grokfast_ma"
)

cd scripts

for exp in "${experiments[@]}"; do
    echo "Testing $exp..."
    if python train_grokk.py --config-path="../config/experiments" --config-name="$exp" train.max_steps=10 > /dev/null 2>&1; then
        echo "✓ $exp configuration works"
    else
        echo "✗ $exp configuration failed"
        exit 1
    fi
done

echo "All configurations tested successfully!"

#!/bin/bash

# Grokking Experiments: L1 Loss and Grokfast Effects
# This script runs a series of experiments to compare the effects of:
# 1. Baseline (no L1, no Grokfast)
# 3. Grokfast EMA only
# 4. Grokfast MA only  

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "scripts/train_grokk.py" ]; then
    print_error "Error: train_grokk.py not found. Please run this script from the grokking repository root."
    exit 1
fi

# Set up Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# Create results directory
mkdir -p results
mkdir -p runs

print_status "Starting Grokking Experiments..."
print_status "This will run 6 experiments comparing L1 loss and Grokfast effects"
print_status "Each experiment will run for 50,000 steps"

# Array of experiment configurations
experiments=(
    "baseline"
    "grokfast_only"
    "grokfast_ma_only"
)

# Array of experiment descriptions
descriptions=(
    "Baseline (No L1, No Grokfast)"
    "Grokfast EMA Only"
    "Grokfast MA Only"
)

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local exp_desc=$2
    local config_path="config/experiments/${exp_name}.yaml"

    print_status "Starting experiment: ${exp_desc}"
    print_status "Config: ${config_path}"

    if [ ! -f "$config_path" ]; then
        print_error "Config file not found: $config_path"
        return 1
    fi

    # Run the experiment (stay in root directory)
    if cd scripts && python train_grokk.py --config-path="../config/experiments" --config-name="$exp_name"; then
        cd ..
        print_success "Completed experiment: ${exp_desc}"
    else
        cd .. 2>/dev/null || true  # Return to root even if cd scripts failed
        print_error "Failed experiment: ${exp_desc}"
        return 1
    fi
    
    # Move tensorboard logs to results directory
    if [ -d "runs/${exp_name}" ]; then
        mv "runs/${exp_name}" "results/${exp_name}_tensorboard"
        print_status "Moved tensorboard logs to results/${exp_name}_tensorboard"
    fi
}

# Main execution
total_experiments=${#experiments[@]}
current_experiment=0

for i in "${!experiments[@]}"; do
    current_experiment=$((i + 1))
    exp_name="${experiments[$i]}"
    exp_desc="${descriptions[$i]}"
    
    print_status "Running experiment ${current_experiment}/${total_experiments}: ${exp_desc}"
    
    if run_experiment "$exp_name" "$exp_desc"; then
        print_success "Experiment ${current_experiment}/${total_experiments} completed successfully"
    else
        print_error "Experiment ${current_experiment}/${total_experiments} failed"
        print_warning "Continuing with remaining experiments..."
    fi
    
    echo ""  # Add spacing between experiments
done

print_success "All experiments completed!"
print_status "Results are available in:"
print_status "  - TensorBoard logs: results/*_tensorboard/"
print_status "  - To view results: tensorboard --logdir results/"

# Generate summary
print_status "Experiment Summary:"
echo "1. Baseline: No regularization techniques"
echo "2. Grokfast EMA: alpha = 0.98, lambda = 2.0"
echo "3. Grokfast MA: window = 100, lambda = 2.0"

print_status "To analyze results, compare validation accuracy curves and grokking onset times"
print_status "Expected observations:"
echo "  - Grokfast should also accelerate grokking"
echo "  - Combined approaches may show fastest grokking"
echo "  - EMA vs MA variants may show different convergence patterns"

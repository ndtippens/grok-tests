#!/bin/bash

# Comprehensive Grokking Experiments Runner
# This script runs all available experiments with proper result management:
# - Baseline experiments
# - Grokfast variants (EMA and MA)
# - Different model architectures (GPT-2, GPT-2 ResV, GPT-2 Meta)
# - Automatic result saving and duplicate prevention

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_info() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "train_grokk.py" ]; then
    print_error "Error: train_grokk.py not found. Please run this script from the grokking repository root."
    exit 1
fi

# Set up Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# Create results directory structure
mkdir -p results

# Function to check if experiment results already exist
check_existing_results() {
    local exp_name=$1
    local result_file="results/${exp_name}/experiment_results.json"
    local tensorboard_dir="results/${exp_name}"

    if [ -f "$result_file" ] || [ -d "$tensorboard_dir" ]; then
        return 0  # Results exist
    else
        return 1  # No results found
    fi
}

print_header "=== COMPREHENSIVE GROKKING EXPERIMENTS ==="
print_status "This script will run all available experiments with automatic result management"
print_status "Features:"
print_info "  ✓ Automatic duplicate detection and skipping"
print_info "  ✓ Structured result saving to results/ directory"
print_info "  ✓ TensorBoard log organization"
print_info "  ✓ JSON result summaries"
print_info "  ✓ Multiple model architectures"
echo ""

# Comprehensive array of experiment configurations
experiments=(
    "baseline"
    "grokfast_only"
    "grokfast_ma_only"
    "gpt2_grokfast_ma"
    "gpt2_resv_grokfast_ma"
    "gpt2_meta_grokfast_ma"
)

# Array of experiment descriptions
descriptions=(
    "Baseline (Original Grokk Model, No Grokfast)"
    "Grokfast EMA (Original Grokk Model)"
    "Grokfast MA (Original Grokk Model)"
    "GPT-2 with Grokfast MA"
    "GPT-2 ResV with Grokfast MA"
    "GPT-2 Meta with Grokfast MA"
)

# Array of model types for organization
model_types=(
    "grokk_model"
    "grokk_model"
    "grokk_model"
    "gpt2"
    "gpt2_resv"
    "gpt2_meta"
)

# Function to save experiment metadata
save_experiment_metadata() {
    local exp_name=$1
    local exp_desc=$2
    local model_type=$3
    local start_time=$4
    local end_time=$5
    local status=$6

    local result_dir="results/${exp_name}"
    mkdir -p "$result_dir"

    cat > "${result_dir}/experiment_metadata.json" << EOF
{
    "experiment_name": "${exp_name}",
    "description": "${exp_desc}",
    "model_type": "${model_type}",
    "start_time": "${start_time}",
    "end_time": "${end_time}",
    "status": "${status}",
    "config_file": "experiments/${exp_name}.yaml",
    "tensorboard_logs": "${result_dir}/tensorboard",
    "generated_by": "run_experiments.sh",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local exp_desc=$2
    local model_type=$3
    local config_path="experiments/${exp_name}.yaml"
    local result_dir="results/${exp_name}"
    local start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    print_status "Starting experiment: ${exp_desc}"
    print_info "Config: ${config_path}"
    print_info "Results will be saved to: ${result_dir}"

    if [ ! -f "$config_path" ]; then
        print_error "Config file not found: $config_path"
        save_experiment_metadata "$exp_name" "$exp_desc" "$model_type" "$start_time" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "failed_config_not_found"
        return 1
    fi

    # Create result directory
    mkdir -p "$result_dir"

    # Set up logging
    local log_file="${result_dir}/training.log"
    local tensorboard_dir="${result_dir}/tensorboard"

    print_info "Training log: ${log_file}"
    print_info "TensorBoard logs: ${tensorboard_dir}"

    # Run the experiment with output redirection and Hydra config to disable outputs dir
    if python train_grokk.py --config-name="$exp_name" tensorboard.log_dir="$tensorboard_dir" hydra.run.dir="$result_dir" hydra.output_subdir=null > "$log_file" 2>&1; then
        local end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        print_success "Completed experiment: ${exp_desc}"
        save_experiment_metadata "$exp_name" "$exp_desc" "$model_type" "$start_time" "$end_time" "completed"

        # Extract final metrics from log if possible
        if grep -q "val.*train.*step" "$log_file"; then
            tail -n 20 "$log_file" | grep "val.*train.*step" | tail -n 1 > "${result_dir}/final_metrics.txt"
            print_info "Final metrics saved to ${result_dir}/final_metrics.txt"
        fi

        return 0
    else
        local end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        print_error "Failed experiment: ${exp_desc}"
        save_experiment_metadata "$exp_name" "$exp_desc" "$model_type" "$start_time" "$end_time" "failed"
        print_warning "Check log file for details: ${log_file}"
        return 1
    fi
}

# Function to generate experiment summary
generate_summary() {
    local summary_file="results/experiment_summary.md"
    print_status "Generating experiment summary: ${summary_file}"

    cat > "$summary_file" << EOF
# Grokking Experiments Summary

Generated on: $(date)

## Experiments Overview

This document summarizes all grokking experiments run with different model architectures and optimization techniques.

### Experiment Categories

1. **Baseline Experiments**: Original grokk model without optimizations
2. **Grokfast Variants**: Testing EMA vs MA gradient filtering
3. **Architecture Comparison**: GPT-2, GPT-2 ResV, GPT-2 Meta with Grokfast MA

### Results Structure

Each experiment creates a directory in \`results/\` with:
- \`experiment_metadata.json\`: Experiment configuration and timing
- \`training.log\`: Complete training output
- \`final_metrics.txt\`: Final validation/training metrics
- \`tensorboard/\`: TensorBoard event files

### Experiments Run

EOF

    # Add experiment details
    for i in "${!experiments[@]}"; do
        local exp_name="${experiments[$i]}"
        local exp_desc="${descriptions[$i]}"
        local model_type="${model_types[$i]}"
        local result_dir="results/${exp_name}"

        echo "#### ${exp_name}" >> "$summary_file"
        echo "- **Description**: ${exp_desc}" >> "$summary_file"
        echo "- **Model**: ${model_type}" >> "$summary_file"

        if [ -f "${result_dir}/experiment_metadata.json" ]; then
            local status=$(grep '"status"' "${result_dir}/experiment_metadata.json" | cut -d'"' -f4)
            echo "- **Status**: ${status}" >> "$summary_file"

            if [ -f "${result_dir}/final_metrics.txt" ]; then
                echo "- **Final Metrics**: \`$(cat "${result_dir}/final_metrics.txt")\`" >> "$summary_file"
            fi
        else
            echo "- **Status**: not_run" >> "$summary_file"
        fi

        echo "" >> "$summary_file"
    done

    cat >> "$summary_file" << EOF

### Analysis Instructions

1. **View TensorBoard logs**:
   \`\`\`bash
   tensorboard --logdir results/
   \`\`\`

2. **Compare experiments**: Look for differences in:
   - Validation accuracy curves
   - Training loss convergence
   - Grokking onset timing
   - Final performance

3. **Expected patterns**:
   - Grokfast should accelerate grokking compared to baseline
   - MA variant may show smoother convergence than EMA
   - Different architectures may have varying grokking behaviors

### Key Metrics to Monitor

- **Validation Accuracy**: Primary indicator of grokking
- **Training Loss**: Should decrease steadily
- **Generalization Gap**: Difference between train and validation performance
- **Grokking Onset**: Step where validation accuracy rapidly improves

EOF

    print_success "Summary generated: ${summary_file}"
}

# Main execution
total_experiments=${#experiments[@]}
completed_experiments=0
skipped_experiments=0
failed_experiments=0

print_header "Starting experiment execution..."
print_status "Total experiments to process: ${total_experiments}"
echo ""

for i in "${!experiments[@]}"; do
    current_experiment=$((i + 1))
    exp_name="${experiments[$i]}"
    exp_desc="${descriptions[$i]}"
    model_type="${model_types[$i]}"

    print_header "=== Experiment ${current_experiment}/${total_experiments}: ${exp_name} ==="
    print_info "Description: ${exp_desc}"
    print_info "Model Type: ${model_type}"

    # Check if results already exist
    if check_existing_results "$exp_name"; then
        print_warning "Results already exist for ${exp_name} - SKIPPING"
        print_info "To re-run, delete: results/${exp_name}/"
        skipped_experiments=$((skipped_experiments + 1))
    else
        print_status "No existing results found - running experiment..."

        if run_experiment "$exp_name" "$exp_desc" "$model_type"; then
            print_success "✓ Experiment ${current_experiment}/${total_experiments} completed successfully"
            completed_experiments=$((completed_experiments + 1))
        else
            print_error "✗ Experiment ${current_experiment}/${total_experiments} failed"
            failed_experiments=$((failed_experiments + 1))
            print_warning "Continuing with remaining experiments..."
        fi
    fi

    echo ""  # Add spacing between experiments
done

# Generate final summary
print_header "=== EXPERIMENT EXECUTION COMPLETE ==="
print_success "Execution Summary:"
print_info "  ✓ Completed: ${completed_experiments}"
print_warning "  ⊘ Skipped: ${skipped_experiments}"
print_error "  ✗ Failed: ${failed_experiments}"
print_status "  📊 Total: ${total_experiments}"
echo ""

generate_summary

print_success "All experiments processed!"
print_status "Results are available in:"
print_info "  📁 Individual results: results/<experiment_name>/"
print_info "  📊 TensorBoard: tensorboard --logdir results/"
print_info "  📋 Summary: results/experiment_summary.md"
echo ""

print_header "Next Steps:"
print_info "1. Review experiment_summary.md for overview"
print_info "2. Launch TensorBoard to compare results visually"
print_info "3. Check individual training logs for detailed analysis"
print_info "4. Re-run failed experiments if needed by deleting their result directories"

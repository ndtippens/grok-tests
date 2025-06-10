#!/usr/bin/env python3
"""
Analyze and visualize results from grokking experiments.
This script reads TensorBoard logs and creates comparison plots.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: tensorboard not installed. Install with: pip install tensorboard")
    sys.exit(1)

def load_tensorboard_data(log_dir: str) -> Dict[str, pd.DataFrame]:
    """Load data from TensorBoard logs."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available scalar tags
    scalar_tags = event_acc.Tags()['scalars']
    
    data = {}
    for tag in scalar_tags:
        scalar_events = event_acc.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        data[tag] = pd.DataFrame({'step': steps, 'value': values})
    
    return data

def find_grokking_point(accuracy_data: pd.DataFrame, threshold: float = 0.99) -> int:
    """Find the step where grokking occurs (validation accuracy crosses threshold)."""
    val_acc = accuracy_data[accuracy_data['value'] >= threshold]
    if len(val_acc) > 0:
        return val_acc.iloc[0]['step']
    return -1  # Grokking not achieved

def analyze_experiment(results_dir: str, experiment_name: str) -> Dict:
    """Analyze a single experiment."""
    log_dir = os.path.join(results_dir, f"{experiment_name}")
    
    if not os.path.exists(log_dir):
        print(f"Warning: Log directory not found: {log_dir}")
        return None
    
    try:
        data = load_tensorboard_data(log_dir)
        
        # Extract key metrics
        train_acc = data.get('Accuracy/train', pd.DataFrame())
        val_acc = data.get('Accuracy/val', pd.DataFrame())
        train_loss = data.get('Loss/train', pd.DataFrame())
        val_loss = data.get('Loss/val', pd.DataFrame())
        
        # Find grokking point
        grokking_step = find_grokking_point(val_acc)
        
        # Calculate final accuracies
        final_train_acc = train_acc['value'].iloc[-1] if len(train_acc) > 0 else 0
        final_val_acc = val_acc['value'].iloc[-1] if len(val_acc) > 0 else 0
        
        return {
            'name': experiment_name,
            'data': data,
            'grokking_step': grokking_step,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    except Exception as e:
        print(f"Error analyzing {experiment_name}: {e}")
        return None

def create_comparison_plots(experiments: List[Dict], output_dir: str):
    """Create comparison plots for all experiments."""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Grokking Experiments: L1 Loss and Grokfast Effects', fontsize=16)
    
    # Plot 1: Validation Accuracy
    ax1 = axes[0, 0]
    for i, exp in enumerate(experiments):
        if exp and len(exp['val_acc']) > 0:
            ax1.plot(exp['val_acc']['step'], exp['val_acc']['value'], 
                    label=exp['name'].replace('_', ' ').title(), 
                    color=colors[i % len(colors)], linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Validation Accuracy Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    ax2 = axes[0, 1]
    for i, exp in enumerate(experiments):
        if exp and len(exp['train_loss']) > 0:
            ax2.plot(exp['train_loss']['step'], exp['train_loss']['value'], 
                    label=exp['name'].replace('_', ' ').title(), 
                    color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Validation Loss
    ax3 = axes[1, 0]
    for i, exp in enumerate(experiments):
        if exp and len(exp['val_loss']) > 0:
            ax3.plot(exp['val_loss']['step'], exp['val_loss']['value'], 
                    label=exp['name'].replace('_', ' ').title(), 
                    color=colors[i % len(colors)], linewidth=2)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Validation Loss Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Grokking Comparison (Bar chart)
    ax4 = axes[1, 1]
    exp_names = []
    grokking_steps = []
    colors_bar = []
    
    for i, exp in enumerate(experiments):
        if exp:
            exp_names.append(exp['name'].replace('_', ' ').title())
            grokking_steps.append(exp['grokking_step'] if exp['grokking_step'] > 0 else 50000)
            colors_bar.append(colors[i % len(colors)])
    
    bars = ax4.bar(exp_names, grokking_steps, color=colors_bar, alpha=0.7)
    ax4.set_ylabel('Steps to Grokking')
    ax4.set_title('Grokking Speed Comparison')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, step in zip(bars, grokking_steps):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{step}' if step < 50000 else 'No Grok',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'grokking_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    # Show the plot
    plt.show()

def generate_summary_table(experiments: List[Dict], output_dir: str):
    """Generate a summary table of results."""
    
    summary_data = []
    for exp in experiments:
        if exp:
            summary_data.append({
                'Experiment': exp['name'].replace('_', ' ').title(),
                'Grokking Step': exp['grokking_step'] if exp['grokking_step'] > 0 else 'Not Achieved',
                'Final Train Acc': f"{exp['final_train_acc']:.4f}",
                'Final Val Acc': f"{exp['final_val_acc']:.4f}",
                'Grokking Speed': 'Fast' if exp['grokking_step'] > 0 and exp['grokking_step'] < 20000 else 
                                'Medium' if exp['grokking_step'] > 0 and exp['grokking_step'] < 35000 else
                                'Slow' if exp['grokking_step'] > 0 else 'None'
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'experiment_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Summary table saved to: {csv_path}")
    
    # Print to console
    print("\nExperiment Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze grokking experiment results')
    parser.add_argument('--results-dir', default='results', 
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', default='analysis', 
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # List of experiments to analyze
    experiment_names = [
        'baseline',
        'l1_only',
        'grokfast_only', 
        'grokfast_ma_only',
        'l1_plus_grokfast',
        'l1_plus_grokfast_ma'
    ]
    
    print("Analyzing grokking experiments...")
    
    # Analyze each experiment
    experiments = []
    for exp_name in experiment_names:
        print(f"Analyzing {exp_name}...")
        result = analyze_experiment(args.results_dir, exp_name)
        experiments.append(result)
    
    # Filter out None results
    valid_experiments = [exp for exp in experiments if exp is not None]
    
    if not valid_experiments:
        print("No valid experiment results found!")
        return
    
    # Create comparison plots
    print("Creating comparison plots...")
    create_comparison_plots(valid_experiments, args.output_dir)
    
    # Generate summary table
    print("Generating summary table...")
    generate_summary_table(valid_experiments, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

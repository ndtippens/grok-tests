#!/usr/bin/env python3
"""
Visualize the experimental design for grokking experiments.
Creates a table showing which techniques are enabled in each experiment.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def create_experiment_design_table():
    """Create a visual table showing the experimental design."""
    
    # Define experiments and their configurations
    experiments = [
        'Baseline',
        'L1 Only', 
        'Grokfast EMA Only',
        'Grokfast MA Only',
        'L1 + Grokfast EMA',
        'L1 + Grokfast MA'
    ]
    
    techniques = ['L1 Regularization', 'Grokfast EMA', 'Grokfast MA']
    
    # Create configuration matrix (1 = enabled, 0 = disabled)
    config_matrix = np.array([
        [0, 0, 0],  # Baseline
        [1, 0, 0],  # L1 Only
        [0, 1, 0],  # Grokfast EMA Only
        [0, 0, 1],  # Grokfast MA Only
        [1, 1, 0],  # L1 + Grokfast EMA
        [1, 0, 1],  # L1 + Grokfast MA
    ])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(config_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(techniques)))
    ax.set_yticks(np.arange(len(experiments)))
    ax.set_xticklabels(techniques)
    ax.set_yticklabels(experiments)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(experiments)):
        for j in range(len(techniques)):
            text = '✓' if config_matrix[i, j] == 1 else '✗'
            color = 'white' if config_matrix[i, j] == 1 else 'black'
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=16, fontweight='bold')
    
    # Add title and labels
    ax.set_title("Grokking Experiments: Technique Combinations", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Optimization Techniques", fontsize=12, fontweight='bold')
    ax.set_ylabel("Experiments", fontsize=12, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(techniques)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(experiments)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Disabled', 'Enabled'])
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('experiment_design.png', dpi=300, bbox_inches='tight')
    print("Experiment design visualization saved to: experiment_design.png")
    
    # Show the plot
    plt.show()
    
    return config_matrix, experiments, techniques

def create_parameter_summary():
    """Create a summary table of key parameters."""
    
    data = {
        'Parameter': [
            'Dataset', 'Training Fraction', 'Model Architecture', 'Hidden Dim', 'Attention Heads',
            'Learning Rate', 'Weight Decay', 'Batch Size', 'Max Steps',
            'L1 Weight (when enabled)', 'Grokfast Alpha (EMA)', 'Grokfast Lambda', 'Grokfast Window (MA)'
        ],
        'Value': [
            'Permutation Group (S_5)', '40%', 'Transformer (2 blocks)', '128', '4',
            '0.001', '1.0', '512', '50,000',
            '0.005', '0.98', '2.0', '100'
        ],
        'Description': [
            'Group operations on 5 elements', 'Fraction of all possible examples for training',
            'Standard transformer with 2 attention blocks', 'Hidden dimension size',
            'Number of attention heads per block', 'AdamW learning rate',
            'L2 regularization strength', 'Training batch size', 'Maximum training steps',
            'L1 regularization coefficient', 'EMA decay factor for Grokfast',
            'Gradient amplification factor', 'Moving average window size'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Experiment Parameters Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('parameter_summary.png', dpi=300, bbox_inches='tight')
    print("Parameter summary saved to: parameter_summary.png")
    
    # Show the plot
    plt.show()

def main():
    """Main function to create all visualizations."""
    print("Creating experiment design visualization...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create experiment design table
    config_matrix, experiments, techniques = create_experiment_design_table()
    
    print("\nCreating parameter summary...")
    
    # Create parameter summary
    create_parameter_summary()
    
    print("\nExperiment Design Summary:")
    print("=" * 50)
    print("This experimental design allows us to study:")
    print("1. Individual effects of L1 regularization")
    print("2. Individual effects of Grokfast (both EMA and MA variants)")
    print("3. Combined effects of L1 + Grokfast")
    print("4. Comparison between EMA and MA variants of Grokfast")
    print("\nExpected outcomes:")
    print("- L1 regularization should accelerate grokking")
    print("- Grokfast should also accelerate grokking")
    print("- Combined approaches may show fastest grokking")
    print("- EMA vs MA may show different convergence patterns")

if __name__ == "__main__":
    main()

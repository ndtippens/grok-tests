# Grokking Experiments: L1 Loss and Grokfast Effects

This directory contains configurations and scripts to run systematic experiments comparing the effects of L1 regularization and Grokfast optimization on grokking behavior.

## Overview

The experiments compare 6 different configurations:

1. **Baseline**: No L1 regularization, No Grokfast
2. **L1 Only**: L1 regularization (weight=0.005), No Grokfast  
3. **Grokfast EMA Only**: No L1, Grokfast with EMA (α=0.98, λ=2.0)
4. **Grokfast MA Only**: No L1, Grokfast with Moving Average (window=100, λ=2.0)
5. **L1 + Grokfast EMA**: Both L1 and Grokfast EMA enabled
6. **L1 + Grokfast MA**: Both L1 and Grokfast MA enabled

## Quick Start

### Running All Experiments

```bash
# Make sure you're in the grokking repository root
./run_experiments.sh
```

This will:
- Run all 6 experiments sequentially
- Each experiment runs for 50,000 steps
- Save TensorBoard logs to `results/` directory
- Print progress and completion status

### Running Individual Experiments

```bash
cd scripts
python train_grokk.py --config-path="../config/experiments" --config-name="baseline"
python train_grokk.py --config-path="../config/experiments" --config-name="l1_only"
# ... etc for other experiments
```

### Analyzing Results

After running experiments:

```bash
# Analyze and create comparison plots
python analyze_results.py

# Or specify custom directories
python analyze_results.py --results-dir results --output-dir analysis
```

This will:
- Load TensorBoard logs from all experiments
- Create comparison plots showing validation accuracy, loss curves, and grokking speed
- Generate a summary table with key metrics
- Save plots and tables to the `analysis/` directory

## Expected Results

Based on the grokking literature, you should observe:

### L1 Regularization Effects
- **Faster Grokking**: L1 regularization should accelerate the onset of grokking
- **Better Generalization**: Models with L1 should achieve higher validation accuracy sooner
- **Smoother Curves**: L1 may lead to more stable training dynamics

### Grokfast Effects  
- **Accelerated Grokking**: Both EMA and MA variants should speed up grokking
- **EMA vs MA**: EMA (exponential moving average) may be more stable than MA (moving average)
- **Gradient Filtering**: Grokfast filters gradients to amplify consistent directions

### Combined Effects
- **Synergistic**: L1 + Grokfast may show the fastest grokking
- **Diminishing Returns**: Benefits may plateau when both techniques are used

## Configuration Details

### Dataset Configuration
All experiments use the same dataset:
- **Task**: Permutation group operations (S_5)
- **Training Fraction**: 40% of all possible examples
- **Vocabulary Size**: 96 tokens

### Model Architecture
- **Type**: Transformer with 2 blocks
- **Hidden Dimension**: 128
- **Attention Heads**: 4
- **Attention Dimension**: 32
- **Feed-forward Dimension**: 512

### Training Configuration
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Weight Decay**: 1.0
- **Batch Size**: 512
- **Max Steps**: 50,000
- **Evaluation**: Every 100 steps

### L1 Regularization
- **Weight**: 0.005 (when enabled)
- **Application**: Applied to all model parameters
- **Normalization**: Scaled by parameter norm

### Grokfast Configuration
- **EMA Variant**: α=0.98, λ=2.0
- **MA Variant**: window=100, λ=2.0
- **Application**: Applied to gradients before optimizer step

## File Structure

```
config/experiments/
├── baseline.yaml              # No L1, No Grokfast
├── l1_only.yaml              # L1 only
├── grokfast_only.yaml        # Grokfast EMA only
├── grokfast_ma_only.yaml     # Grokfast MA only
├── l1_plus_grokfast.yaml     # L1 + Grokfast EMA
└── l1_plus_grokfast_ma.yaml  # L1 + Grokfast MA

run_experiments.sh            # Main experiment runner
analyze_results.py           # Results analysis script
EXPERIMENTS_README.md        # This file
```

## Monitoring Progress

### TensorBoard
Each experiment saves logs to `runs/{experiment_name}/`. You can monitor progress with:

```bash
tensorboard --logdir runs/
```

### Console Output
The training script prints metrics every 100 steps:
- Training and validation loss
- Training and validation accuracy  
- Learning rate
- L1 loss (when applicable)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in experiment configs
2. **Slow Training**: Experiments are designed to run for 50k steps (~2-4 hours each)
3. **Missing Dependencies**: Install requirements with `pip install -r requirements.txt`

### Modifying Experiments

To create custom experiments:
1. Copy an existing config from `config/experiments/`
2. Modify the parameters you want to test
3. Update the experiment name and log directory
4. Run with the new config name

### Hardware Requirements

- **GPU**: Recommended for reasonable training times
- **Memory**: ~4GB GPU memory should be sufficient
- **Storage**: ~1GB for all experiment logs and results

## Citation

If you use these experiments in your research, please cite the original grokking paper:

```bibtex
@article{power2022grokking,
  title={Grokking: Generalization beyond overfitting on small algorithmic datasets},
  author={Power, Alethea and Burda, Yuri and Edwards, Harri and Babuschkin, Igor and Misra, Vedant},
  journal={arXiv preprint arXiv:2201.02177},
  year={2022}
}
```

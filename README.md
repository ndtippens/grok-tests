# GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS

An unofficial re-implementation of [this paper](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) by Power et al., with additional model architectures and optimization techniques.

## Project Structure

```
grokking/
├── data/                    # Dataset implementations and configurations
│   ├── datasets.py         # Dataset classes (ModSum, ModDivision, etc.)
│   └── dataset/           # Dataset configuration files
├── models/                 # Model architectures
│   ├── grokk_model.py     # Original Grokking transformer
│   ├── transformer.py     # Base transformer implementation
│   ├── gpt2.py           # GPT-2 architecture
│   ├── gpt2_resv.py      # ResFormer/SVFormer variants
│   ├── gpt2_meta.py      # Meta-learning GPT-2
│   ├── attention.py      # Attention mechanisms
│   ├── positionals.py    # Positional encodings
│   └── resv_attention.py # ResV attention
├── experiments/           # Experiment configurations
│   ├── baseline.yaml     # Baseline experiments
│   ├── grokfast_*.yaml   # Grokfast experiments
│   ├── train_grokk.yaml  # Main training config
│   └── model/           # Model configuration files
├── utils/                # Utility functions
│   ├── utils.py         # General utilities
│   ├── grokfast.py      # Grokfast optimization
│   ├── load_objs.py     # Object loading utilities
│   ├── BitLinear.py     # BitLinear implementation
│   └── muon.py          # Muon optimizer
├── train_grokk.py        # Main training script
├── run_grokfast_ma_experiments.py  # Experiment runner
└── analyze_results.py    # Results analysis
```

## Installation

```bash
git clone https://github.com/Sea-Snell/grokking.git
cd grokking/
pip install -r requirements.txt
```

## Quick Start

To reproduce the original grokking results:

```bash
python train_grokk.py
```

To run all experiments with comprehensive result management:

```bash
./run_experiments.sh
```

This will automatically:
- Run all available experiments (baseline, Grokfast variants, different architectures)
- Skip experiments that already have results
- Save structured results to `results/` directory
- Generate experiment summary and TensorBoard logs

![](grokk.png)
*Running the above commands should give curves like this.*

## Features

### Model Architectures
- **Original Grokking Model**: The transformer from the original paper
- **GPT-2**: Standard GPT-2 architecture with causal attention
- **GPT-2 ResV**: ResFormer/SVFormer with value residual connections
- **GPT-2 Meta**: Meta-learning variant with cross-attention

### Optimization Techniques
- **Grokfast**: Exponential moving average gradient filtering
- **Grokfast-MA**: Moving average variant with configurable window size
- **L1 Regularization**: Weight decay with L1 penalty
- **AdamW**: Standard optimizer with weight decay

### Datasets
- **Modular Arithmetic**: Addition, subtraction, division
- **Permutation Groups**: Group operations on permutations
- **Variable Binding**: Sequence-to-sequence variable assignments

## Configuration

The project uses [Hydra](https://hydra.cc/docs/intro) for configuration management. Configurations are organized in the `experiments/` directory:

- `experiments/train_grokk.yaml`: Main training configuration
- `experiments/baseline.yaml`: Baseline without optimizations
- `experiments/grokfast_*.yaml`: Various Grokfast experiments
- `experiments/model/`: Model-specific configurations
- `data/dataset/`: Dataset configurations

### Running Custom Experiments

```bash
# Run with specific experiment config
python train_grokk.py --config-name=grokfast_ma_only

# Override parameters
python train_grokk.py train.max_steps=5000 train.lr=0.0001

# Use different model
python train_grokk.py model=gpt2_meta
```

## Monitoring

Training supports both TensorBoard and Weights & Biases:

- **TensorBoard**: Logs are saved to `results/` directory structure
- **Weights & Biases**: Set `wandb.use_wandb=true` in config or via command line

## Results Analysis

### Automated Experiment Management

The `run_experiments.sh` script provides comprehensive experiment management:

```bash
# Run all experiments with duplicate detection
./run_experiments.sh

# View experiment summary
cat results/experiment_summary.md

# Launch TensorBoard to compare results
tensorboard --logdir results/
```

### Result Structure

Each experiment creates a structured result directory:
```
results/<experiment_name>/
├── experiment_metadata.json  # Experiment configuration and timing
├── training.log              # Complete training output
├── final_metrics.txt         # Final validation/training metrics
└── tensorboard/             # TensorBoard event files
```

### Key Features

- **Duplicate Prevention**: Automatically skips experiments with existing results
- **Structured Logging**: Organized results with metadata and metrics
- **TensorBoard Integration**: Automatic log organization for visualization
- **Summary Generation**: Markdown summary of all experiments
- **Failure Handling**: Graceful handling of failed experiments with detailed logs

## Contributing

When adding new models or datasets:

1. Add model implementations to `models/`
2. Register new models in `utils/load_objs.py`
3. Add dataset implementations to `data/datasets.py`
4. Create configuration files in appropriate directories
5. Update this README with new features

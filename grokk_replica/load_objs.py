import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from grokk_replica.datasets import ModSumDataset, ModSubtractDataset, ModDivisonDataset, PermutationGroup
from grokk_replica.grokk_model import GrokkModel
from grokk_replica.utils import convert_path
registry = {}

def register(name):
    def add_f(f):
        registry[name] = f
        return f
    return add_f

def load_item(config, *args, verbose=True):
    config = config.copy()
    name = config.pop('name')
    if name not in registry:
        raise NotImplementedError
    if verbose:
        print(f'loading {name}: {config}')
    return registry[name](config, *args, verbose=verbose)

@register('mod_sum_dataset')
def load_mod_sum_dataset(config, verbose=True):
    return ModSumDataset(config['p'], config['frac_train'])

@register('mod_subtract_dataset')
def load_mod_subtract_dataset(config, verbose=True):
    return ModSubtractDataset(config['p'], config['frac_train'])

@register('mod_division_dataset')
def load_mod_division_dataset(config, verbose=True):
    return ModDivisonDataset(config['p'], config['frac_train'])

@register('permutation_group_dataset')
def load_permutation_group_dataset(config, verbose=True):
    return PermutationGroup(config['k'], config['frac_train'])

@register('grokk_model')
def load_grokk_model(config, vocab_size, out_size, device, verbose=True):
    # Extract l1_weight from config, defaulting to 0.0 if not present
    l1_weight = config.get('l1_weight', 0.0)
    # Copy transformer_config and ensure l1_weight is not present
    transformer_config = dict(config['transformer_config'])
    #transformer_config.pop('l1_weight', None)
    # Create model with l1_weight
    model = GrokkModel(transformer_config, vocab_size, out_size, device, l1_weight=l1_weight).to(device)
    # Optionally load checkpoint
    checkpoint_path = config.get('checkpoint_path')
    if checkpoint_path is not None:
        checkpoint_path_converted = convert_path(checkpoint_path)
        if checkpoint_path_converted is not None:
            if verbose:
                print(f'loading grokk_model state dict from: {checkpoint_path_converted}')
            model.load_state_dict(
                torch.load(checkpoint_path_converted, map_location='cpu'),
                strict=config.get('strict_load', True)
            )
            if verbose:
                print('loaded.')
    return model

def register_model_loaders():
    from models.gpt2 import GPT2Model
    from models.gpt2_resv import GPT2ResVModel
    from models.gpt2_meta import GPT2MetaModel
    from models.tokenformer import Tokenformer

    @register('gpt2_model')
    def load_gpt2_model(config, vocab_size, out_size, device, verbose=True):
        model = GPT2Model(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 768),
            num_heads=config.get('num_heads', 12),
            num_layers=config.get('num_layers', 12),
            d_ff=config.get('d_ff', 3072),
            max_seq_len=config.get('max_seq_len', 1024),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        return model

    @register('gpt2_resv_model')
    def load_gpt2_resv_model(config, vocab_size, out_size, device, verbose=True):
        model = GPT2ResVModel(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 768),
            num_heads=config.get('num_heads', 12),
            num_layers=config.get('num_layers', 12),
            d_ff=config.get('d_ff', 3072),
            max_seq_len=config.get('max_seq_len', 1024),
            dropout=config.get('dropout', 0.1),
            share_values=config.get('share_values', False)
        ).to(device)
        return model

    # Add more loaders as needed for meta and tokenformer

register_model_loaders()


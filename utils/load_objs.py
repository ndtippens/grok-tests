import torch
from data.datasets import ModSumDataset, ModSubtractDataset, ModDivisonDataset, PermutationGroup, VarBindingDataset
from models.grokk_model import GrokkModel
from utils.utils import convert_path
from utils.splus import SPlus
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

@register('varbinding_dataset')
def load_varbinding_dataset(config, verbose=True):
    return VarBindingDataset(config['csv_path'], config['frac_train'])

@register('grokk_model')
def load_grokk_model(config, vocab_size, out_size, device, verbose=True):
    transformer_config = config['transformer_config']
    model = GrokkModel(transformer_config, vocab_size, out_size, device).to(device)
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

# Import model classes
from models.gpt2 import GPT2Model
from models.gpt2_resv import GPT2ResVModel
from models.gpt2_meta import GPT2MetaModel

@register('gpt2')
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

@register('gpt2_resv')
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

@register('gpt2_meta')
def load_gpt2_meta_model(config, vocab_size, out_size, device, verbose=True):
    model = GPT2MetaModel(
        vocab_size=vocab_size,
        d_model=config.get('d_model', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 12),
        d_ff=config.get('d_ff', 3072),
        max_seq_len=config.get('max_seq_len', 1024),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    return model

# Optimizer registration functions
@register('adamw')
def load_adamw_optimizer(config, model_parameters, verbose=True):
    return torch.optim.AdamW(
        model_parameters,
        lr=config.get('lr', 0.001),
        weight_decay=config.get('weight_decay', 0.01),
        betas=config.get('betas', [0.9, 0.999])
    )

@register('splus')
def load_splus_optimizer(config, model_parameters, verbose=True):
    return SPlus(
        model_parameters,
        lr=config.get('lr', 0.1),
        b1=config.get('b1', 0.9),
        b2=config.get('b2', 0.999),
        weight_decay=config.get('weight_decay', 0.01),
        ema_rate=config.get('ema_rate', 0.999),
        inverse_every=config.get('inverse_every', 100),
        eps=config.get('eps', 1e-30),
        max_dim=config.get('max_dim', 10000),
        nonstandard_constant=config.get('nonstandard_constant', 0.001)
    )


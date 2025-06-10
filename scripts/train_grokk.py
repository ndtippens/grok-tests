import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grokk_replica.datasets import AbstractDataset
from grokk_replica.utils import combine_logs, parameter_norm
from grokk_replica.load_objs import load_item
from torch.utils.data import IterableDataset, DataLoader
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from grokk_replica.grokfast import gradfilter_ema, gradfilter_ma
import torch.nn.functional as F

def grokk_loss_fn(model, x, y):
    predictions, attns = model(x)
    ce_loss = F.cross_entropy(predictions[:, -1, :], y)
    param_norm = parameter_norm(model)
    l1_loss = 0.0
    if hasattr(model, 'l1_weight') and model.l1_weight > 0:
        for param in model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        l1_loss *= model.l1_weight / param_norm
    loss = ce_loss + l1_loss
    accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == y).float().mean()
    attn_entropies = sum([-(attn * torch.log(attn+1e-7)).sum(dim=-1).mean().item() for attn in attns]) / len(attns)
    return loss, {
        'loss': (loss.item(), x.shape[0]),
        'ce_loss': (ce_loss.item(), x.shape[0]),
        'l1_loss': (l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss, 1),
        'accuracy': (accuracy.item(), x.shape[0]),
        'attn_entropy': (attn_entropies, len(attns)*x.shape[0]*(x.shape[1]-1)),
        'param_norm': (param_norm, 1)
    }

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def train(config):
    print('using config:', config)
    train_cfg = config['train']
    
    # Setup TensorBoard writer
    use_tensorboard = config.get('tensorboard', {}).get('use_tensorboard', True)
    log_dir = config.get('tensorboard', {}).get('log_dir', 'runs')
    writer = SummaryWriter(log_dir=log_dir) if use_tensorboard else None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], 
                              betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1))

    # Initialize grads for Grokfast based on type
    use_grokfast = train_cfg.get('use_grokfast', True)
    grokfast_type = train_cfg.get('grokfast_type', 'ema')  # 'ema' or 'ma'
    grads_ema = None  # For EMA grokfast
    grads_ma = None   # For MA grokfast
    step = 0
    for x, y in tqdm(train_dataloader):
        loss, logs = grokk_loss_fn(model, x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()  # Calculate the gradients.

        # Apply Grokfast if enabled
        if use_grokfast:
            if grokfast_type == 'ema':
                # Option 1: Grokfast EMA (has argument alpha, lamb)
                alpha = train_cfg.get('grokfast_alpha', 0.98)
                lamb = train_cfg.get('grokfast_lamb', 2.0)
                grads_ema = gradfilter_ema(model, grads=grads_ema, alpha=alpha, lamb=lamb)
            elif grokfast_type == 'ma':
                # Option 2: Grokfast-MA (has argument window_size, lamb)
                window_size = train_cfg.get('grokfast_window_size', 100)
                lamb = train_cfg.get('grokfast_lamb', 2.0)
                grads_ma = gradfilter_ma(model, grads=grads_ma, window_size=window_size, lamb=lamb)
        optim.step()  # Call the optimizer.
        lr_schedule.step()
        if (step+1) % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                for i, (val_x, val_y) in tqdm(enumerate(val_dataloader)):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = grokk_loss_fn(model, val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
            val_combined = combine_logs(all_val_logs)
            train_combined = combine_logs([logs])
            out_log = {'val': val_combined, 'train': train_combined, 'step': (step+1), 
                       'lr': float(lr_schedule.get_last_lr()[0])}
            print(out_log)
            
            # Log metrics to TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_combined['loss'], step+1)
                writer.add_scalar('Loss/val', val_combined['loss'], step+1)
                writer.add_scalar('Accuracy/train', train_combined['accuracy'], step+1)
                writer.add_scalar('Accuracy/val', val_combined['accuracy'], step+1)
                writer.add_scalar('Learning_rate', float(lr_schedule.get_last_lr()[0]), step+1)
                
                # Log other metrics
                for k in train_combined:
                    if k not in ['loss', 'accuracy']:
                        writer.add_scalar(f'Metrics_train/{k}', train_combined[k], step+1)
                for k in val_combined:
                    if k not in ['loss', 'accuracy']:
                        writer.add_scalar(f'Metrics_val/{k}', val_combined[k], step+1)
            
            model.train()
        step += 1
        if train_cfg['max_steps'] is not None and step >= train_cfg['max_steps']:
            break
    
    if writer:
        writer.close()


@hydra.main(config_path="../config", config_name="train_grokk", version_base="1.2")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()


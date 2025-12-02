import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import argparse
import os
from tqdm import tqdm
import json
from contextlib import nullcontext

from models import GATLSTM, GATGRU, PlainLSTM
from utils import load_metr_la, compute_all_metrics, compute_metrics_per_horizon


def train_epoch(model, train_loader, optimizer, criterion, device, edge_index, edge_attr, scaler, use_amp=False, amp_dtype=torch.float16, grad_scaler=None):
    model.train()
    total_loss = 0
    num_batches = 0
    
    autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
    for x, y in tqdm(train_loader, desc="Training"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast_ctx:
            out = model(x, edge_index, edge_attr)
            y_pred = out.transpose(1, 2)
            loss = criterion(y_pred, y)
        
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, data_loader, device, edge_index, edge_attr, scaler, use_amp=False, amp_dtype=torch.float16):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext()
        for x, y in tqdm(data_loader, desc="Evaluating"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with autocast_ctx:
                out = model(x, edge_index, edge_attr)
                y_pred = out.transpose(1, 2)
            
            y_pred_rescaled = scaler.inverse_transform(y_pred.detach().float().cpu().numpy())
            y_rescaled = scaler.inverse_transform(y.detach().float().cpu().numpy())
            
            all_preds.append(y_pred_rescaled)
            all_labels.append(y_rescaled)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    all_preds = torch.FloatTensor(all_preds)
    all_labels = torch.FloatTensor(all_labels)
    
    metrics = compute_metrics_per_horizon(all_preds, all_labels, horizons=[3, 6, 12])
    
    return metrics


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        try:
            print(f"CUDA device name: {torch.cuda.get_device_name(device.index)}")
        except Exception:
            pass
    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Prefer bf16 on Ampere+ if available
            bf16_ok = hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        use_amp = True
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        grad_scaler = None if bf16_ok else torch.cuda.amp.GradScaler()
    else:
        use_amp = False
        amp_dtype = torch.float16
        grad_scaler = None
    
    print("Loading METR-LA dataset...")
    train_loader, val_loader, test_loader, edge_index, edge_attr, scaler = load_metr_la(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        max_timesteps=args.max_timesteps
    )
    
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    
    sample_x, sample_y = next(iter(train_loader))
    node_feat_dim = sample_x.shape[-1]
    edge_feat_dim = edge_attr.shape[-1]
    num_nodes = sample_x.shape[2]
    
    print(f"Dataset info:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Node features: {node_feat_dim}")
    print(f"  Edge features: {edge_feat_dim}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    if args.model == 'gatlstm':
        model = GATLSTM(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.pred_len,
            heads=args.heads,
            dropout=args.dropout,
            num_nodes=num_nodes,
            use_per_node_init=args.use_per_node_init,
            use_temporal_attention=args.use_temporal_attention,
            use_skip_connection=args.use_skip_connection,
            use_edge_mlp=args.use_edge_mlp,
            vectorized=args.vectorized
        ).to(device)
    elif args.model == 'gatgru':
        model = GATGRU(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.pred_len,
            heads=args.heads,
            dropout=args.dropout
        ).to(device)
    elif args.model == 'plainlstm':
        model = PlainLSTM(
            node_feat_dim=node_feat_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.pred_len,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"\nModel: {args.model.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.L1Loss()
    try:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {
        'train_loss': [],
        'val_metrics': [],
        'best_epoch': 0
    }
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, edge_index, edge_attr, scaler, use_amp=use_amp, amp_dtype=amp_dtype, grad_scaler=grad_scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_metrics = evaluate(model, val_loader, device, edge_index, edge_attr, scaler, use_amp=use_amp, amp_dtype=amp_dtype)
        val_mae = val_metrics['overall']['mae']

        print(f"Val MAE: {val_mae:.4f}, RMSE: {val_metrics['overall']['rmse']:.4f}")
        print(f"  Horizon 3 (15min):  MAE={val_metrics['horizon_3']['mae']:.4f}")
        print(f"  Horizon 6 (30min):  MAE={val_metrics['horizon_6']['mae']:.4f}")
        print(f"  Horizon 12 (60min): MAE={val_metrics['horizon_12']['mae']:.4f}")
        
        results['train_loss'].append(train_loss)
        results['val_metrics'].append(val_metrics)
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            results['best_epoch'] = epoch
            
            save_path = os.path.join(args.save_dir, f'best_{args.model}_metrla.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'args': vars(args)
            }, save_path)
            print(f"Best model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("\n" + "="*50)
    print("Loading best model and evaluating on test set...")
    checkpoint = torch.load(os.path.join(args.save_dir, f'best_{args.model}_metrla.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, edge_index, edge_attr, scaler)
    
    print("\nTest Results:")
    print(f"Overall - MAE: {test_metrics['overall']['mae']:.4f}, RMSE: {test_metrics['overall']['rmse']:.4f}")
    print(f"Horizon 3 (15min)  - MAE: {test_metrics['horizon_3']['mae']:.4f}, RMSE: {test_metrics['horizon_3']['rmse']:.4f}")
    print(f"Horizon 6 (30min)  - MAE: {test_metrics['horizon_6']['mae']:.4f}, RMSE: {test_metrics['horizon_6']['rmse']:.4f}")
    print(f"Horizon 12 (60min) - MAE: {test_metrics['horizon_12']['mae']:.4f}, RMSE: {test_metrics['horizon_12']['rmse']:.4f}")
    
    results['test_metrics'] = test_metrics
    
    results_path = os.path.join(args.save_dir, f'{args.model}_metrla_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAT-LSTM/GRU on METR-LA')
    
    parser.add_argument('--model', type=str, default='gatlstm', choices=['gatlstm', 'gatgru', 'plainlstm'],
                        help='Model architecture')
    parser.add_argument('--data_dir', type=str, default='./data/METR-LA',
                        help='Directory containing METR-LA data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save models')
    
    parser.add_argument('--seq_len', type=int, default=12,
                        help='Input sequence length (default: 12 = 1 hour)')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Prediction horizon (default: 12 = 1 hour)')
    
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers for plainlstm')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--early_stop', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--max_timesteps', type=int, default=None,
                        help='Optional limit on number of time steps to load (useful for quicker runs)')
    
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_per_node_init', action='store_true',
                        help='Use per-node learnable h0/c0 initial states')
    parser.add_argument('--use_temporal_attention', action='store_true',
                        help='Use temporal attention readout over hidden states')
    parser.add_argument('--use_skip_connection', action='store_true',
                        help='Add input→output skip from last frame')
    parser.add_argument('--use_edge_mlp', action='store_true',
                        help='Apply tiny MLP to edge_attr before GAT')
    parser.add_argument('--vectorized', action='store_true',
                        help='Enable batch-parallel vectorized GAT-LSTM over disjoint union graph')
    
    args = parser.parse_args()
    main(args)

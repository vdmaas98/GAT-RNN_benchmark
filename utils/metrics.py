import torch
import numpy as np


def masked_mae(preds, labels, null_val=np.nan):
    """
    Masked Mean Absolute Error
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    """
    Masked Root Mean Squared Error
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def masked_mape(preds, labels, null_val=np.nan, epsilon=1e-5):
    """
    Masked Mean Absolute Percentage Error
    """
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / (labels + epsilon))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def compute_all_metrics(preds, labels, null_val=np.nan):
    """
    Compute MAE, RMSE, and MAPE
    
    Args:
        preds: [batch_size, pred_len, num_nodes] or [batch_size, num_nodes, pred_len]
        labels: same shape as preds
    
    Returns:
        dict with 'mae', 'rmse', 'mape' keys
    """
    mae = masked_mae(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def compute_metrics_per_horizon(preds, labels, horizons=[3, 6, 12], null_val=np.nan):
    """
    Compute metrics at specific prediction horizons
    
    Args:
        preds: [batch_size, pred_len, num_nodes]
        labels: [batch_size, pred_len, num_nodes]
        horizons: list of time steps to evaluate (1-indexed, e.g., [3, 6, 12] for 15min, 30min, 60min)
    
    Returns:
        dict with horizon-specific metrics
    """
    results = {}
    
    for h in horizons:
        if h <= preds.shape[1]:
            h_idx = h - 1
            h_preds = preds[:, h_idx, :]
            h_labels = labels[:, h_idx, :]
            
            metrics = compute_all_metrics(h_preds, h_labels, null_val)
            results[f'horizon_{h}'] = metrics
    
    overall = compute_all_metrics(preds, labels, null_val)
    results['overall'] = overall
    
    return results

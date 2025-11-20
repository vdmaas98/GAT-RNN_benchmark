import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class METRLA_Dataset(Dataset):
    def __init__(self, data, edge_index, edge_attr, seq_len=12, pred_len=12, split='train'):
        """
        Args:
            data: numpy array [num_timesteps, num_nodes, num_features]
            edge_index: torch tensor [2, num_edges]
            edge_attr: torch tensor [num_edges, edge_feat_dim]
            seq_len: input sequence length
            pred_len: prediction horizon
            split: 'train', 'val', or 'test'
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
        num_samples = data.shape[0]
        num_train = int(num_samples * 0.7)
        num_val = int(num_samples * 0.1)
        
        if split == 'train':
            self.data = data[:num_train]
        elif split == 'val':
            self.data = data[num_train:num_train + num_val]
        else:
            self.data = data[num_train + num_val:]
        
        self.num_samples = len(self.data) - seq_len - pred_len + 1
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, :, 0]
        
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        
        return x, y


def load_metr_la(data_dir, seq_len=12, pred_len=12, batch_size=64):
    """
    Load METR-LA dataset
    
    Expected files in data_dir:
        - metr-la.h5: traffic speed data
        - adj_mx.pkl: adjacency matrix
        - distances_la_2012.csv: distance matrix for edge features
    
    Returns:
        train_loader, val_loader, test_loader, edge_index, edge_attr, scaler
    """
    
    data_path = os.path.join(data_dir, 'metr-la.h5')
    adj_path = os.path.join(data_dir, 'adj_mx.pkl')
    
    df = pd.read_hdf(data_path)
    data = df.values

    max_timesteps = 5000
    data = data[:max_timesteps]
    
    num_nodes = data.shape[1]

    scaler = StandardScaler(mean=data.mean(), std=data.std())
    data = scaler.transform(data)
    
    with open(adj_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    
    edge_index, edge_attr = adj_matrix_to_edge_index(adj_mx)
    
    data = data.reshape(data.shape[0], num_nodes, -1)
    if data.shape[2] == 1:
        time_of_day = np.array([i % 288 for i in range(data.shape[0])]) / 288.0
        day_of_week = np.array([i // 288 % 7 for i in range(data.shape[0])]) / 7.0
        
        time_of_day = np.tile(time_of_day[:, None, None], (1, num_nodes, 1))
        day_of_week = np.tile(day_of_week[:, None, None], (1, num_nodes, 1))
        
        data = np.concatenate([data, time_of_day, day_of_week], axis=-1)
    
    train_dataset = METRLA_Dataset(data, edge_index, edge_attr, seq_len, pred_len, 'train')
    val_dataset = METRLA_Dataset(data, edge_index, edge_attr, seq_len, pred_len, 'val')
    test_dataset = METRLA_Dataset(data, edge_index, edge_attr, seq_len, pred_len, 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, edge_index, edge_attr, scaler


def adj_matrix_to_edge_index(adj_mx, threshold=0.1):
    """
    Convert adjacency matrix to edge_index and edge_attr
    
    Args:
        adj_mx: numpy array [num_nodes, num_nodes]
        threshold: minimum weight to include edge
    
    Returns:
        edge_index: torch tensor [2, num_edges]
        edge_attr: torch tensor [num_edges, 1]
    """
    adj_mx = adj_mx.copy()
    adj_mx[adj_mx < threshold] = 0
    
    edge_index = []
    edge_attr = []
    
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if adj_mx[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append([adj_mx[i, j]])
    
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    
    return edge_index, edge_attr


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean

import torch
import torch.nn as nn


class PlainLSTM(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=node_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_seq):
        b, t, n, f = x_seq.shape
        x_flat = x_seq.view(b * n, t, f)

        lstm_out, _ = self.lstm(x_flat)
        h_last = lstm_out[:, -1, :]

        out_flat = self.output(h_last)
        out = out_flat.view(b, n, self.output_dim)
        return out

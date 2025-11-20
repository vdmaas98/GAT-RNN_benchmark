import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, heads=2, dropout=0.3):
        super().__init__()
        self.gat_x_r = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_r = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_x_u = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_u = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_x_c = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_c = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)

        self.ln = nn.LayerNorm(hidden_channels)

    def forward(self, x_t, h_prev, edge_index, edge_attr, return_attn=False):
        if return_attn:
            x_r, (attn_edge_index, attn_alpha) = self.gat_x_r(x_t, edge_index, edge_attr, return_attention_weights=True)
        else:
            x_r = self.gat_x_r(x_t, edge_index, edge_attr)
            attn_edge_index, attn_alpha = None, None

        r_t = torch.sigmoid(x_r + self.gat_h_r(h_prev, edge_index, edge_attr))
        u_t = torch.sigmoid(self.gat_x_u(x_t, edge_index, edge_attr) + self.gat_h_u(h_prev, edge_index, edge_attr))
        h_hat_t = torch.tanh(self.gat_x_c(x_t, edge_index, edge_attr) + r_t * self.gat_h_c(h_prev, edge_index, edge_attr))
        h_t = u_t * h_prev + (1 - u_t) * h_hat_t

        h_t = self.ln(h_t)

        if return_attn:
            return h_t, (attn_edge_index, attn_alpha)
        return h_t


class GATGRU(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, output_dim=12, heads=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.cell = GATGRUCell(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            edge_dim=edge_feat_dim,
            heads=heads,
            dropout=dropout
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_seq, edge_index, edge_attr, return_attn=False):
        """
        Args:
            x_seq: [batch_size, seq_len, num_nodes, node_feat_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_feat_dim]
            return_attn: bool
        
        Returns:
            out: [batch_size, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, _ = x_seq.shape
        device = x_seq.device

        all_outputs = []
        all_attn_weights = []

        for b in range(batch_size):
            h = torch.zeros(num_nodes, self.hidden_dim, device=device)

            sample_attn_weights = []

            for t in range(seq_len):
                x_t = x_seq[b, t]

                if return_attn:
                    h, attn_weights = self.cell(x_t, h, edge_index, edge_attr, return_attn=True)
                    sample_attn_weights.append(attn_weights)
                else:
                    h = self.cell(x_t, h, edge_index, edge_attr, return_attn=False)

            out_b = self.output(h)
            all_outputs.append(out_b)

            if return_attn:
                all_attn_weights.append(sample_attn_weights)

        outputs = torch.stack(all_outputs)

        if return_attn:
            return outputs, all_attn_weights
        return outputs

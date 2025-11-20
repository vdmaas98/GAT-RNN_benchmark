import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, heads=2, dropout=0.3, use_residual=False):
        super().__init__()
        self.use_residual = use_residual

        self.gat_x_i = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_i = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_x_f = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_f = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_x_o = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_o = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_x_g = GATv2Conv(in_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat_h_g = GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, edge_dim=edge_dim, dropout=dropout)

        self.ln = nn.LayerNorm(hidden_channels)

        if self.use_residual:
            self.proj_res = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_t, h_prev, c_prev, edge_index, edge_attr, return_attn=False):
        if return_attn:
            i_x, attn_weights = self.gat_x_i(x_t, edge_index, edge_attr, return_attention_weights=True)
        else:
            i_x = self.gat_x_i(x_t, edge_index, edge_attr)
            attn_weights = None

        i_h = self.gat_h_i(h_prev, edge_index, edge_attr)
        i_t = torch.sigmoid(i_x + i_h)

        f_t = torch.sigmoid(self.gat_x_f(x_t, edge_index, edge_attr) + self.gat_h_f(h_prev, edge_index, edge_attr))
        o_t = torch.sigmoid(self.gat_x_o(x_t, edge_index, edge_attr) + self.gat_h_o(h_prev, edge_index, edge_attr))
        g_t = torch.tanh(self.gat_x_g(x_t, edge_index, edge_attr) + self.gat_h_g(h_prev, edge_index, edge_attr))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        if self.use_residual:
            h_t = h_t + self.proj_res(h_prev)

        h_t = self.ln(h_t)
        return h_t, c_t, attn_weights


class GATLSTM(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, output_dim=12, heads=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.cell = GATLSTMCell(
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
            c = torch.zeros(num_nodes, self.hidden_dim, device=device)

            batch_attn_weights = []
            for t in range(seq_len):
                x_t = x_seq[b, t]

                h, c, attn = self.cell(x_t, h, c, edge_index, edge_attr, return_attn)
                if return_attn:
                    batch_attn_weights.append(attn)

            out_b = self.output(h)
            all_outputs.append(out_b)
            if return_attn:
                all_attn_weights.append(batch_attn_weights)

        if return_attn:
            return torch.stack(all_outputs), all_attn_weights
        return torch.stack(all_outputs)

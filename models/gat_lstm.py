import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, heads=2, dropout=0.3, use_residual=False, use_edge_mlp=False):
        super().__init__()
        self.use_residual = use_residual
        self.use_edge_mlp = use_edge_mlp

        if self.use_edge_mlp:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim)
            )

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
        edge_attr_proc = self.edge_mlp(edge_attr) if (edge_attr is not None and getattr(self, 'edge_mlp', None) is not None) else edge_attr
        if return_attn:
            i_x, attn_weights = self.gat_x_i(x_t, edge_index, edge_attr_proc, return_attention_weights=True)
        else:
            i_x = self.gat_x_i(x_t, edge_index, edge_attr_proc)
            attn_weights = None

        i_h = self.gat_h_i(h_prev, edge_index, edge_attr_proc)
        i_t = torch.sigmoid(i_x + i_h)

        f_t = torch.sigmoid(self.gat_x_f(x_t, edge_index, edge_attr_proc) + self.gat_h_f(h_prev, edge_index, edge_attr_proc))
        o_t = torch.sigmoid(self.gat_x_o(x_t, edge_index, edge_attr_proc) + self.gat_h_o(h_prev, edge_index, edge_attr_proc))
        g_t = torch.tanh(self.gat_x_g(x_t, edge_index, edge_attr_proc) + self.gat_h_g(h_prev, edge_index, edge_attr_proc))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        if self.use_residual:
            h_t = h_t + self.proj_res(h_prev)

        h_t = self.ln(h_t)
        return h_t, c_t, attn_weights


class GATLSTM(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, output_dim=12, heads=2, dropout=0.3, num_nodes=None,
                 use_per_node_init=False, use_temporal_attention=False, use_skip_connection=False, use_edge_mlp=False,
                 vectorized=False, use_horizon_embeddings=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.use_per_node_init = use_per_node_init
        self.use_temporal_attention = use_temporal_attention
        self.use_skip_connection = use_skip_connection
        self.vectorized = vectorized
        self.use_horizon_embeddings = use_horizon_embeddings

        self.cell = GATLSTMCell(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            edge_dim=edge_feat_dim,
            heads=heads,
            dropout=dropout,
            use_edge_mlp=use_edge_mlp
        )

        if self.use_per_node_init:
            if self.num_nodes is None:
                raise ValueError("num_nodes must be provided when use_per_node_init=True")
            self.h0 = nn.Parameter(torch.zeros(self.num_nodes, hidden_dim))
            self.c0 = nn.Parameter(torch.zeros(self.num_nodes, hidden_dim))

        if self.use_temporal_attention:
            self.attn_q = nn.Linear(hidden_dim, hidden_dim)
            self.attn_k = nn.Linear(hidden_dim, hidden_dim)
            self.attn_v = nn.Linear(hidden_dim, hidden_dim)

        if self.use_horizon_embeddings:
            self.horizon_emb = nn.Embedding(output_dim, hidden_dim)
            self.output_per_step = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        if self.use_skip_connection:
            self.skip = nn.Linear(node_feat_dim, output_dim)

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

        if not self.vectorized:
            for b in range(batch_size):
                if self.use_per_node_init:
                    h = self.h0
                    c = self.c0
                else:
                    h = torch.zeros(num_nodes, self.hidden_dim, device=device)
                    c = torch.zeros(num_nodes, self.hidden_dim, device=device)

                batch_attn_weights = []
                h_states = []
                for t in range(seq_len):
                    x_t = x_seq[b, t]
                    h, c, attn = self.cell(x_t, h, c, edge_index, edge_attr, return_attn)
                    h_states.append(h)
                    if return_attn:
                        batch_attn_weights.append(attn)

                if self.use_temporal_attention:
                    H = torch.stack(h_states, dim=0)
                    H_n = H.permute(1, 0, 2)
                    q = self.attn_q(h)
                    K = self.attn_k(H_n)
                    V = self.attn_v(H_n)
                    scores = torch.matmul(q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)
                    alpha = torch.softmax(scores, dim=-1)
                    h_agg = torch.matmul(alpha.unsqueeze(1), V).squeeze(1)
                else:
                    h_agg = h

                if self.use_horizon_embeddings:
                    idx = torch.arange(self.output_dim, device=device)
                    e = self.horizon_emb(idx)
                    Hn = h_agg.unsqueeze(1) + e.unsqueeze(0)
                    out_b = self.output_per_step(Hn).squeeze(-1)
                else:
                    out_b = self.output(h_agg)
                if self.use_skip_connection:
                    x_last = x_seq[b, -1]
                    out_b = out_b + self.skip(x_last)
                all_outputs.append(out_b)
                if return_attn:
                    all_attn_weights.append(batch_attn_weights)
        else:
            B = batch_size
            N = num_nodes
            E = edge_index.size(1)
            edge_index_b = edge_index.repeat(1, B)
            offset = (torch.arange(B, device=device) * N).repeat_interleave(E)
            offset = offset.to(edge_index_b.dtype)
            edge_index_b = edge_index_b + offset.unsqueeze(0)
            if edge_attr is not None:
                edge_attr_b = edge_attr.repeat(B, 1)
            else:
                edge_attr_b = None

            if self.use_per_node_init:
                h = self.h0.repeat(B, 1)
                c = self.c0.repeat(B, 1)
            else:
                h = torch.zeros(B * N, self.hidden_dim, device=device)
                c = torch.zeros(B * N, self.hidden_dim, device=device)

            h_states = []
            for t in range(seq_len):
                x_t = x_seq[:, t].reshape(B * N, -1)
                h, c, _ = self.cell(x_t, h, c, edge_index_b, edge_attr_b, return_attn=False)
                h_states.append(h)

            if self.use_temporal_attention:
                H = torch.stack(h_states, dim=0)
                H_bn = H.permute(1, 0, 2)
                q = self.attn_q(h)
                K = self.attn_k(H_bn)
                V = self.attn_v(H_bn)
                scores = torch.matmul(q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / (self.hidden_dim ** 0.5)
                alpha = torch.softmax(scores, dim=-1)
                h_agg = torch.matmul(alpha.unsqueeze(1), V).squeeze(1)
            else:
                h_agg = h

            if self.use_horizon_embeddings:
                idx = torch.arange(self.output_dim, device=device)
                e = self.horizon_emb(idx)
                Hbn = h_agg.view(B, N, -1)
                Hbn = Hbn.unsqueeze(2) + e.view(1, 1, self.output_dim, -1)
                out = self.output_per_step(Hbn).squeeze(-1)
            else:
                out = self.output(h_agg)
            if self.use_skip_connection:
                x_last = x_seq[:, -1].reshape(B, N, -1)
                out = out + self.skip(x_last)
            out = out.view(B, N, self.output_dim)
            all_outputs = out

        if return_attn:
            if isinstance(all_outputs, list):
                return torch.stack(all_outputs), all_attn_weights
            else:
                return all_outputs, all_attn_weights
        else:
            if isinstance(all_outputs, list):
                return torch.stack(all_outputs)
            else:
                return all_outputs

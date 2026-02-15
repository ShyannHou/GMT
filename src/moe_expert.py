
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from math import sqrt

class GraphAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):

        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = sqrt(self.d_k)

    def forward(self, g, h):

        N = h.shape[0]
        H = self.n_heads

        q = self.W_Q(h).view(N, H, self.d_k)
        k = self.W_K(h).view(N, H, self.d_k)
        v = self.W_V(h).view(N, H, self.d_k)

        g.ndata['q'] = q
        g.ndata['k'] = k
        g.ndata['v'] = v

        g.apply_edges(fn.v_dot_u('q', 'k', 'score'))
        g.edata['score'] = g.edata['score'] / self.scale

        g.edata['alpha'] = dgl.ops.edge_softmax(g, g.edata['score'])
        g.edata['alpha'] = self.dropout(g.edata['alpha'])

        g.apply_edges(fn.copy_u('v', 'msg'))
        g.edata['msg'] = g.edata['msg'] * g.edata['alpha']
        g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'att_out'))

        att_out = g.ndata['att_out'].view(N, -1)
        out = self.out_proj(att_out)

        g.ndata.pop('q')
        g.ndata.pop('k')
        g.ndata.pop('v')
        g.ndata.pop('att_out')

        return out

class ExpertLayer(nn.Module):

    def __init__(self, d_model, n_heads=4, d_ff=None, dropout=0.1):

        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = GraphAttention(d_model, n_heads, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):

        att_out = self.attention(g, h)

        h_res = h + self.dropout(att_out)

        out = self.mlp(self.norm(h_res))
        return out

class DyMoEGNNLayer(nn.Module):

    def __init__(self, d_model, num_experts, n_heads=4, d_ff=None,
                 dropout=0.1, top_k=2):

        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            ExpertLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, g, h):

        N, d = h.shape

        gate_logits = self.gate(h)

        gate_scores, selected = torch.topk(gate_logits, self.top_k, dim=-1)

        gate_weights = F.softmax(gate_scores, dim=-1)

        expert_outputs = torch.stack(
            [expert(g, h) for expert in self.experts], dim=1
        )

        selected_expanded = selected.unsqueeze(-1).expand(N, self.top_k, d)
        selected_outputs = torch.gather(expert_outputs, 1, selected_expanded)

        out = (gate_weights.unsqueeze(-1) * selected_outputs).sum(dim=1)
        return out

class MoEGNN(nn.Module):

    def __init__(self, gnn_backbone, d_hidden, output_dim,
                 num_experts=4, n_heads=4, top_k=2, dropout=0.1):

        super().__init__()

        self.gnn = gnn_backbone

        self.moe = DyMoEGNNLayer(
            d_model=d_hidden,
            num_experts=num_experts,
            n_heads=n_heads,
            top_k=top_k,
            dropout=dropout,
        )

        self.output_proj = nn.Linear(d_hidden, output_dim)

        self.return_hidden = False

    def forward(self, g, features):

        self.gnn.return_hidden = True
        h = self.gnn(g, features)
        self.gnn.return_hidden = False

        h_moe = self.moe(g, h)

        if self.return_hidden:
            return h_moe

        out = self.output_proj(h_moe)
        return out

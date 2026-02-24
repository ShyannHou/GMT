
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from math import sqrt

from moe_attention import GraphMoESelfAttention

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
    """
    Strict Eq.(10) MoE wrapper.

    Each expert follows:
      f_v = MLP(h_v + Att(h_v, {h_u | u in N(v)}))

    Attention implementation is reused from moe_attention.AttentionLayer flow
    through GraphMoESelfAttention/GraphAttentionExpert.
    """

    def __init__(self, d_model, num_experts, n_heads=4, d_ff=None,
                 dropout=0.1, top_k=2):

        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Use Eq.(10)/(11)-style expert implementation.
        self.strict_moe = GraphMoESelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_experts=num_experts,
            mlp_hidden=d_ff,
            dropout=dropout,
            include_self=True,
            use_topology=False,
            topk_experts=top_k,
        )

        self.last_gate_prob = None
        self.last_gate_logits = None

    def forward(self, g, h):
        out, gate_prob, gate_logits, _ = self.strict_moe(g, h, node_blocks=None)
        self.last_gate_prob = gate_prob
        self.last_gate_logits = gate_logits
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


class ParallelGNNMoE(nn.Module):
    """
    Parallel at the SAME representation level:
      Branch-1: identity branch of GNN hidden h
      Branch-2: Eq.(10) MoE branch over the same h

    Then fuse [h, h_moe] and feed MLP classifier.

    Optional topology feature hook (safe by default):
      - set use_topology_feature=True
      - attach per-node topology features as g.ndata['topo_feat']
      - model will add a projected topology residual before final classifier
      - with topo_trainable_fusion=True, fusion weight is learned
    """

    def __init__(self, gnn_backbone, input_dim, d_hidden, output_dim,
                 num_experts=4, n_heads=4, top_k=2, dropout=0.1,
                 use_topology_feature=False, topo_alpha=0.1,
                 topo_trainable_fusion=False):
        super().__init__()

        self.gnn = gnn_backbone
        self.use_topology_feature = use_topology_feature
        self.topo_alpha = topo_alpha
        self.topo_trainable_fusion = topo_trainable_fusion

        self.moe = DyMoEGNNLayer(
            d_model=d_hidden,
            num_experts=num_experts,
            n_heads=n_heads,
            top_k=top_k,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, output_dim),
        )

        if self.use_topology_feature:
            # Keep parameters initialized (avoid LazyLinear checkpoint edge cases).
            self.topo_proj = nn.Linear(d_hidden, d_hidden)
            self.topo_norm = nn.LayerNorm(d_hidden)
            if self.topo_trainable_fusion:
                self.topo_gate = nn.Sequential(
                    nn.Linear(2 * d_hidden, d_hidden),
                    nn.ReLU(),
                    nn.Linear(d_hidden, 1),
                )
                self.topo_alpha_logit = nn.Parameter(torch.tensor(float(topo_alpha)))

        self.return_hidden = False

    def forward(self, g, features, return_parts: bool = False):
        """Forward.

        If return_parts=True, additionally return (h, h_moe, topo_h) for optional
        modality-alignment losses (e.g., MMD) without affecting default behavior.
        """
        topo_h = None

        # Shared base representation h (same-level input for both branches)
        self.gnn.return_hidden = True
        h = self.gnn(g, features)
        self.gnn.return_hidden = False

        # Eq.(10) MoE branch on the same h
        h_moe = self.moe(g, h)

        # Late fusion + MLP
        h_fused = self.fusion(torch.cat([h, h_moe], dim=-1))

        # Optional topology residual (no-op unless enabled and feature exists)
        if self.use_topology_feature and isinstance(g, dgl.DGLGraph) and ('topo_feat' in g.ndata):
            topo_feat = g.ndata['topo_feat'].float()
            d = h_fused.shape[-1]
            if topo_feat.shape[-1] > d:
                topo_feat = topo_feat[:, :d]
            elif topo_feat.shape[-1] < d:
                pad = torch.zeros(topo_feat.shape[0], d - topo_feat.shape[-1], device=topo_feat.device, dtype=topo_feat.dtype)
                topo_feat = torch.cat([topo_feat, pad], dim=-1)
            topo_h = self.topo_norm(self.topo_proj(topo_feat))
            if self.topo_trainable_fusion:
                gate = torch.sigmoid(self.topo_gate(torch.cat([h_fused, topo_h], dim=-1)))
                alpha = torch.sigmoid(self.topo_alpha_logit)
                h_fused = h_fused + alpha * gate * topo_h
            else:
                h_fused = h_fused + self.topo_alpha * topo_h

        if self.return_hidden:
            if return_parts:
                return h_fused, h, h_moe, topo_h
            return h_fused

        out = self.classifier(h_fused)
        if return_parts:
            return out, h, h_moe, topo_h
        return out

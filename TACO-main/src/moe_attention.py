import math
from typing import List, Optional, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    """
    Reusable scaled dot-product attention (adapted from TimeVLM SelfAttention_Family.py).

    Input shapes:
      queries: [B, L, H, E]
      keys:    [B, S, H, E]
      values:  [B, S, H, D]
      attn_bias: optional additive score bias [B, H, L, S]
    Output:
      out:     [B, L, H, D]
      attn:    [B, H, L, S]
    """

    def __init__(self, scale: Optional[float] = None, attention_dropout: float = 0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, attn_bias=None):
        _, _, _, e = queries.shape
        scale = self.scale or (1.0 / math.sqrt(e))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if attn_bias is not None:
            scores = scores + attn_bias
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        return out.contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Same interface as TimeVLM's AttentionLayer, with optional attn_bias.

    Input:
      queries: [B, L, d_model]
      keys:    [B, S, d_model]
      values:  [B, S, d_model]
    Output:
      out:     [B, L, d_model]
      attn:    [B, H, L, S]
    """

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys: Optional[int] = None, d_values: Optional[int] = None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, attn_bias=None):
        bsz, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        h = self.n_heads

        queries = self.query_projection(queries).view(bsz, q_len, h, -1)
        keys = self.key_projection(keys).view(bsz, k_len, h, -1)
        values = self.value_projection(values).view(bsz, k_len, h, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, attn_bias=attn_bias)
        out = out.view(bsz, q_len, -1)
        return self.out_projection(out), attn


class GraphAttentionExpert(nn.Module):
    """
    One expert implementing Eq.(10)-(11) style node update:
      f_v = MLP(h_v + Att(h_v, {h_u | u in N(v)}))

    Default path is vectorized over edges with DGL ops (fast), while preserving
    Eq.(10)/(11) math. A reference per-node path is kept for debugging.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.1,
        include_self: bool = True,
        fast_mode: bool = True,
    ):
        super().__init__()
        self.include_self = include_self
        self.fast_mode = fast_mode

        self.attn = AttentionLayer(
            attention=FullAttention(attention_dropout=dropout),
            d_model=d_model,
            n_heads=n_heads,
        )

        hidden = mlp_hidden or (2 * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def _incoming_neighbors(self, g: dgl.DGLGraph) -> List[List[int]]:
        src, dst = g.edges()
        n = g.num_nodes()
        neigh = [[] for _ in range(n)]
        src_list = src.tolist()
        dst_list = dst.tolist()
        for s, d in zip(src_list, dst_list):
            neigh[d].append(s)
        if self.include_self:
            for v in range(n):
                neigh[v].append(v)
        return neigh

    def _forward_reference(self, g: dgl.DGLGraph, h: torch.Tensor, expert_beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Readable reference implementation (slow)."""
        neigh = self._incoming_neighbors(g)
        n, _ = h.shape
        out = torch.zeros_like(h)

        for v in range(n):
            idx = neigh[v]
            if len(idx) == 0:
                idx = [v]

            idx_t = torch.tensor(idx, device=h.device, dtype=torch.long)
            q = h[v : v + 1].unsqueeze(0)       # [1, 1, D]
            kv = h[idx_t].unsqueeze(0)          # [1, S, D]

            attn_bias = None
            if expert_beta is not None:
                # Eq.(13): add log(beta_t) on neighbors as topology-aware score bias
                b = expert_beta[idx_t].clamp(min=1e-8)
                attn_bias = torch.log(b).view(1, 1, 1, -1)

            # Eq.(11): Att(h, U) = softmax(qK^T/sqrt(d))V
            att_out, _ = self.attn(q, kv, kv, attn_mask=None, attn_bias=attn_bias)  # [1, 1, D]
            att_v = att_out.squeeze(0).squeeze(0)                                     # [D]

            # Eq.(10): f_v = MLP(h_v + Att(...))
            out[v] = self.mlp((h[v] + att_v).unsqueeze(0)).squeeze(0)

        return out

    def _forward_vectorized(self, g: dgl.DGLGraph, h: torch.Tensor, expert_beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Vectorized Eq.(10)/(11) path with DGL edge softmax.
        For each edge u->v:
          score(u,v) = <q_v, k_u>/sqrt(d)
          alpha(u,v) = softmax_u(score(u,v))
          Att(v) = sum_u alpha(u,v) * v_u
          out(v) = MLP(h_v + Att(v))
        """
        n = h.shape[0]
        h_heads = self.attn.n_heads

        # Reuse AttentionLayer projections (teacher-requested AttentionLayer flow)
        q = self.attn.query_projection(h).view(n, h_heads, -1)
        k = self.attn.key_projection(h).view(n, h_heads, -1)
        v = self.attn.value_projection(h).view(n, h_heads, -1)

        scale = self.attn.inner_attention.scale
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])

        with g.local_scope():
            g.ndata["q"] = q
            g.ndata["k"] = k
            g.ndata["v"] = v

            # score: q_v dot k_u
            g.apply_edges(fn.v_dot_u("q", "k", "score"))
            score = g.edata["score"]
            if score.dim() == 3 and score.size(-1) == 1:
                score = score.squeeze(-1)  # [E, H]

            if expert_beta is not None:
                g.ndata["beta"] = expert_beta.clamp(min=1e-8)
                g.apply_edges(fn.copy_u("beta", "beta_src"))
                beta_log = torch.log(g.edata["beta_src"]).unsqueeze(-1)  # [E, 1]
                score = score + beta_log

            score = score * scale

            alpha = dgl.ops.edge_softmax(g, score)
            alpha = self.attn.inner_attention.dropout(alpha)

            g.apply_edges(fn.copy_u("v", "v_src"))
            g.edata["msg"] = g.edata["v_src"] * alpha.unsqueeze(-1)
            g.update_all(fn.copy_e("msg", "m"), fn.sum("m", "att_out"))

            att = g.ndata["att_out"].reshape(n, -1)
            att = self.attn.out_projection(att)

        # Eq.(10): f_v = MLP(h_v + Att(...))
        return self.mlp(h + att)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, expert_beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.fast_mode:
            return self._forward_vectorized(g, h, expert_beta=expert_beta)
        return self._forward_reference(g, h, expert_beta=expert_beta)


class GraphMoESelfAttention(nn.Module):
    """
    Mixture-of-Experts wrapper over multiple GraphAttentionExpert blocks.

    expert_out_i = Expert_i(g, h)
    gate = softmax(W_g h)
    h_out = sum_i gate_i * expert_out_i

    Optional topology-aware learning (Eq.12-13):
      beta_{u,t} = sigmoid(p_t · (W^P h_u))
      attention score += log(beta_{u,t})
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        num_experts: int = 4,
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.1,
        include_self: bool = True,
        use_topology: bool = False,
        hard_block_mask: bool = True,
        topk_experts: Optional[int] = None,
        fast_mode: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_topology = use_topology
        self.hard_block_mask = hard_block_mask
        self.topk_experts = topk_experts
        self.fast_mode = fast_mode

        self.experts = nn.ModuleList(
            [
                GraphAttentionExpert(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    include_self=include_self,
                    fast_mode=fast_mode,
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(d_model, num_experts)

        if use_topology:
            self.topo_proj = nn.Linear(d_model, d_model, bias=False)  # W^P
            self.topo_gates = nn.Parameter(torch.randn(num_experts, d_model))  # p_t

    def _compute_gate_prob(self, h: torch.Tensor, active_experts: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(h)  # [N, E]

        # Dynamic expert growth behavior: only first active_experts are enabled.
        if active_experts is not None and active_experts < self.num_experts:
            gate_logits[:, active_experts:] = float("-inf")

        if self.topk_experts is not None and self.topk_experts < self.num_experts:
            k = max(1, int(self.topk_experts))
            topk_idx = torch.topk(gate_logits, k=k, dim=-1).indices
            keep = torch.zeros_like(gate_logits, dtype=torch.bool)
            keep.scatter_(1, topk_idx, True)
            gate_logits = gate_logits.masked_fill(~keep, float("-inf"))
        gate_prob = torch.softmax(gate_logits, dim=-1)
        return gate_logits, gate_prob

    def _compute_beta(self, h: torch.Tensor, node_blocks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_topology:
            return None
        proj = self.topo_proj(h)  # [N, D]
        beta_all = torch.sigmoid(torch.matmul(proj, self.topo_gates.t()))  # [N, E]

        # Optional hard prior: expert t should suppress nodes from future blocks (> t)
        if self.hard_block_mask and node_blocks is not None:
            exp_ids = torch.arange(1, self.num_experts + 1, device=h.device).view(1, -1)  # [1, E]
            allow = (node_blocks.view(-1, 1) <= exp_ids).float()  # [N, E]
            beta_all = beta_all * allow + (1.0 - allow) * 1e-8
        return beta_all

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, node_blocks: Optional[torch.Tensor] = None):
        # Dynamic active expert count from visible block ids (if available)
        active_experts = None
        if node_blocks is not None:
            active_experts = int(torch.clamp(node_blocks.max(), min=1, max=self.num_experts).item())

        # gate
        gate_logits, gate_prob = self._compute_gate_prob(h, active_experts=active_experts)  # [N, E]

        # topology-aware beta for each (node, expert)
        beta_all = self._compute_beta(h, node_blocks=node_blocks)  # [N, E] or None

        # expert outputs
        expert_outs = []
        for e, expert in enumerate(self.experts):
            expert_beta = None if beta_all is None else beta_all[:, e]
            expert_outs.append(expert(g, h, expert_beta=expert_beta))
        all_expert_out = torch.stack(expert_outs, dim=1)  # [N, E, D]

        mixed = (all_expert_out * gate_prob.unsqueeze(-1)).sum(1)   # [N, D]
        return mixed, gate_prob, gate_logits, beta_all


def block_guided_loss(gate_logits: torch.Tensor, node_blocks: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Eq.(8): CE over expert selection by block index.

    gate_logits: [N, E]
    node_blocks: [N]  (1-based block index)
    """
    target = (node_blocks - 1).clamp(min=0, max=num_experts - 1).long()
    return F.cross_entropy(gate_logits, target)


def graph_block_guided_loss(beta_all: torch.Tensor, node_blocks: torch.Tensor) -> torch.Tensor:
    """
    Eq.(14): BCE over topology-aware node/expert compatibility.

    beta_all:    [N, E], beta_{u,t}
    node_blocks: [N], 1-based block index b(u)
    label l_{u,t} = 1 if t >= b(u) else 0
    """
    n, e = beta_all.shape
    exp_ids = torch.arange(1, e + 1, device=beta_all.device).view(1, -1)
    labels = (exp_ids >= node_blocks.view(-1, 1)).float()
    return F.binary_cross_entropy(beta_all, labels)

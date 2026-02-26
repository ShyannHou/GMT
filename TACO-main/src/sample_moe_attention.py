"""
Sample code for teacher discussion:
- Get GNN output H
- Feed H into AttentionLayer-based MoE self-attention (Eq.10 / Eq.11 style)

Run (from src/):
  python sample_moe_attention.py --dataset PEMS04 --task 0 --hidden_dim 48
"""

import argparse
import pickle

import torch

from models import GCN
from moe_attention import GraphMoESelfAttention, block_guided_loss, graph_block_guided_loss


def main(args):
    data_path = f"../data/{args.dataset}/"
    with open(data_path + f"sub_graph_{args.task}_by_edges", "rb") as f:
        g = pickle.load(f)

    x = g.ndata["x"].float()
    in_dim = x.shape[1]

    # Build 1-based block ids from node_idxs/new_nodes_mask if available.
    if "block_id" not in g.ndata:
        if "node_idxs" in g.ndata and "new_nodes_mask" in g.ndata:
            # Fallback heuristic for demo: old nodes -> 1, current new nodes -> task+1
            block_id = torch.ones(g.num_nodes(), dtype=torch.long)
            new_idx = g.ndata["new_nodes_mask"].nonzero().view(-1)
            block_id[new_idx] = args.task + 1
            g.ndata["block_id"] = block_id
        else:
            g.ndata["block_id"] = torch.ones(g.num_nodes(), dtype=torch.long)

    # 1) GNN output H
    backbone = GCN(in_dim, args.hidden_dim, args.hidden_dim)
    backbone.return_hidden = True
    with torch.no_grad():
        H = backbone(g, x)  # [N, hidden_dim]

    # 2) Feed H to MoE self-attention (query=node, key/value=neighbors)
    moe_attn = GraphMoESelfAttention(
        d_model=args.hidden_dim,
        n_heads=args.moe_heads,
        num_experts=args.moe_experts,
        dropout=args.dropout,
        use_topology=args.use_topology,
        topk_experts=args.moe_topk,
    )

    node_blocks = g.ndata["block_id"] if "block_id" in g.ndata else None
    with torch.no_grad():
        H_new, gate_prob, gate_logits, beta_all = moe_attn(g, H, node_blocks=node_blocks)

    print("Graph nodes:", g.num_nodes())
    print("H shape:", tuple(H.shape))
    print("H_new shape:", tuple(H_new.shape))
    print("Gate shape:", tuple(gate_prob.shape), "(N, num_experts)")
    print("Example gate_prob[0]:", gate_prob[0].tolist())

    if "block_id" in g.ndata:
        block_ids = g.ndata["block_id"].long()
        bl = block_guided_loss(gate_logits, block_ids, num_experts=args.moe_experts)
        print("Block-guided loss (Eq.8):", float(bl))
        if beta_all is not None:
            gbl = graph_block_guided_loss(beta_all, block_ids)
            print("Graph block-guided loss (Eq.14):", float(gbl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PEMS04")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--moe_experts", type=int, default=4)
    parser.add_argument("--moe_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_topology", action="store_true")
    parser.add_argument("--moe_topk", type=int, default=None)
    args = parser.parse_args()
    main(args)

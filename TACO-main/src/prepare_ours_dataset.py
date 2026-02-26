import os
import pickle
import random
from typing import List, Tuple

import dgl
import numpy as np
import torch
import torch.nn.functional as F


def parse_graph_adj_npz(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    arr = np.load(path, allow_pickle=True)
    indptr = arr["indptr"].astype(np.int64)
    indices = arr["indices"].astype(np.int64)
    shape = arr["shape"].astype(np.int64)
    n = int(shape[0])

    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    dst = indices
    return src, dst, n


def stratified_masks(labels: torch.Tensor, train_ratio=0.3, val_ratio=0.2, seed=0):
    rng = random.Random(seed)
    n = labels.shape[0]
    train_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    classes = sorted(labels.unique().tolist())
    for c in classes:
        idx = (labels == c).nonzero().view(-1).tolist()
        rng.shuffle(idx)
        m = len(idx)
        m_train = int(round(m * train_ratio))
        m_val = int(round(m * val_ratio))

        train_ids = idx[:m_train]
        val_ids = idx[m_train:m_train + m_val]
        test_ids = idx[m_train + m_val:]

        train_mask[torch.tensor(train_ids, dtype=torch.long)] = True
        valid_mask[torch.tensor(val_ids, dtype=torch.long)] = True
        test_mask[torch.tensor(test_ids, dtype=torch.long)] = True

    return train_mask, valid_mask, test_mask


def build_slot_graph(dropbox_dir: str, slot: int, graphs_per_slot: int, global_node_offset: int):
    ids = list(range(slot * graphs_per_slot, (slot + 1) * graphs_per_slot))

    src_all = []
    dst_all = []
    x_all = []
    y_all = []
    node_idxs_all = []

    local_offset = 0

    for prefix, glabel in [("G1", 0), ("G2", 1)]:
        for idx in ids:
            base = os.path.join(dropbox_dir, f"{prefix}_graph_{idx}")
            adj_path = base + "_adj.npz"
            node_path = base + "_node_labels.npy"
            label_path = base + "_graph_label.npy"

            if not (os.path.exists(adj_path) and os.path.exists(node_path) and os.path.exists(label_path)):
                raise FileNotFoundError(f"Missing file for {base}")

            src, dst, n = parse_graph_adj_npz(adj_path)
            src_all.append(src + local_offset)
            dst_all.append(dst + local_offset)

            node_labels = np.load(node_path, allow_pickle=True).astype(np.int64)
            x = F.one_hot(torch.from_numpy(node_labels), num_classes=10).float()

            graph_label = int(np.load(label_path, allow_pickle=True))
            # Fallback to folder label if file is malformed
            if graph_label not in [0, 1]:
                graph_label = glabel

            y = torch.full((n,), graph_label, dtype=torch.int64)
            node_idxs = torch.arange(global_node_offset, global_node_offset + n, dtype=torch.int64)

            x_all.append(x)
            y_all.append(y)
            node_idxs_all.append(node_idxs)

            global_node_offset += n
            local_offset += n

    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    num_nodes = local_offset

    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=num_nodes)

    x_cat = torch.cat(x_all, dim=0)
    y_cat = torch.cat(y_all, dim=0)
    node_idxs_cat = torch.cat(node_idxs_all, dim=0)

    g.ndata["x"] = x_cat
    g.ndata["y"] = y_cat
    g.ndata["node_idxs"] = node_idxs_cat
    g.ndata["new_nodes_mask"] = torch.ones(num_nodes, dtype=torch.int32)
    g.ndata["num_new_nodes"] = torch.full((num_nodes,), num_nodes, dtype=torch.int64)
    g.ndata["_ID"] = torch.arange(num_nodes, dtype=torch.int64)

    return g, global_node_offset


def main():
    root = "/root/kunlin/Shyann-Research/TACO-main"
    dropbox_dir = os.path.join(root, "dropbox_import_20260219")
    out_dir = os.path.join(root, "data", "OURS")

    os.makedirs(out_dir, exist_ok=True)

    num_task = 10
    num_class = 2
    graphs_per_slot = 10

    g_list: List[dgl.DGLGraph] = []
    global_node_offset = 0
    for slot in range(num_task):
        g, global_node_offset = build_slot_graph(dropbox_dir, slot, graphs_per_slot, global_node_offset)
        g_list.append(g)

        with open(os.path.join(out_dir, f"sub_graph_{slot}_by_edges"), "wb") as f:
            pickle.dump(g, f)
        with open(os.path.join(out_dir, f"graph_{slot}_by_edges"), "wb") as f:
            pickle.dump(g, f)

    with open(os.path.join(out_dir, "statistics"), "wb") as f:
        pickle.dump((num_task, num_class), f)

    for run in range(10):
        masks = []
        for slot, g in enumerate(g_list):
            train_mask, valid_mask, test_mask = stratified_masks(
                g.ndata["y"],
                train_ratio=0.3,
                val_ratio=0.2,
                seed=run * 100 + slot,
            )
            masks.append((train_mask, valid_mask, test_mask))

        with open(os.path.join(out_dir, f"mask_seed_{run}"), "wb") as f:
            pickle.dump(masks, f)

    print(f"Created dataset at: {out_dir}")
    print(f"num_task={num_task}, num_class={num_class}, total_global_nodes={global_node_offset}")


if __name__ == "__main__":
    main()

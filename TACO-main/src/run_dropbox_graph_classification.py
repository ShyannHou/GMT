import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader

from models import GCN
from moe_expert import MoEGNN, ParallelGNNMoE
from pd_image_model import CNN, ExpandMLP


@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]


class GraphDataset(Dataset):
    def __init__(
        self,
        graphs: List[dgl.DGLGraph],
        labels: List[int],
        indices: List[int],
        topo_images: Optional[torch.Tensor] = None,
    ):
        self.graphs = [graphs[i] for i in indices]
        self.labels = torch.tensor([labels[i] for i in indices], dtype=torch.long)
        self.topo_images = None if topo_images is None else topo_images[indices]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.topo_images is None:
            return self.graphs[idx], self.labels[idx]
        return self.graphs[idx], self.labels[idx], self.topo_images[idx]


class GraphClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        mode: str,
        num_experts: int,
        moe_n_heads: int,
        moe_top_k: int,
        dropout: float,
        use_topology_feature: bool = False,
        topo_alpha: float = 1.0,
    ):
        super().__init__()
        backbone = GCN(input_dim, hidden_dim, hidden_dim)

        if mode == "parallel":
            self.encoder = ParallelGNNMoE(
                backbone,
                input_dim,
                hidden_dim,
                hidden_dim,
                num_experts=num_experts,
                n_heads=moe_n_heads,
                top_k=moe_top_k,
                dropout=dropout,
            )
        elif mode == "serial":
            self.encoder = MoEGNN(
                backbone,
                hidden_dim,
                hidden_dim,
                num_experts=num_experts,
                n_heads=moe_n_heads,
                top_k=moe_top_k,
                dropout=dropout,
            )
        elif mode == "gnn":
            self.encoder = backbone
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.encoder.return_hidden = True
        self.pool = dgl.nn.AvgPooling()

        self.use_topology_feature = use_topology_feature
        self.topo_alpha = topo_alpha
        if use_topology_feature:
            # Topology branch: PD -> CNN -> ExpandMLP (n aligned to num_experts)
            self.topo_cnn = CNN(input_channel=2, dim_out=hidden_dim)
            self.topo_expand = ExpandMLP(d=hidden_dim, n=num_experts)
            self.topo_gate = nn.Linear(hidden_dim, num_experts)

            self.classifier = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, bg: dgl.DGLGraph, topo_img: Optional[torch.Tensor] = None):
        x = bg.ndata["x"].float()
        h = self.encoder(bg, x)
        hg = self.pool(bg, h)

        if self.use_topology_feature:
            if topo_img is None:
                raise ValueError("topo_img is required when use_topology_feature=True")
            t = self.topo_cnn(topo_img.float())                 # [B, D]
            t_tokens = self.topo_expand(t)                      # [B, E, D]
            w = torch.softmax(self.topo_gate(hg), dim=-1)       # [B, E]
            t_agg = (t_tokens * w.unsqueeze(-1)).sum(dim=1)     # [B, D]
            t_agg = self.topo_alpha * t_agg
            z = torch.cat([hg, t_agg], dim=-1)                  # [B, 2D]
            return self.classifier(z)

        return self.classifier(hg)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_graph_adj_npz(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    arr = np.load(path, allow_pickle=True)
    indptr = arr["indptr"].astype(np.int64)
    indices = arr["indices"].astype(np.int64)
    shape = arr["shape"].astype(np.int64)
    n = int(shape[0])

    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    dst = indices
    return src, dst, n


def load_dropbox_graphs(data_dir: str) -> Tuple[List[dgl.DGLGraph], List[int]]:
    graphs, labels = [], []

    for prefix in ["G1", "G2"]:
        for idx in range(100):
            base = os.path.join(data_dir, f"{prefix}_graph_{idx}")
            adj_path = base + "_adj.npz"
            node_path = base + "_node_labels.npy"
            glabel_path = base + "_graph_label.npy"

            if not (os.path.exists(adj_path) and os.path.exists(node_path) and os.path.exists(glabel_path)):
                raise FileNotFoundError(f"Missing files for {base}")

            src, dst, n = parse_graph_adj_npz(adj_path)
            g = dgl.graph((src, dst), num_nodes=n)
            g = dgl.add_self_loop(g)

            node_labels = np.load(node_path, allow_pickle=True).astype(np.int64)
            x = F.one_hot(torch.from_numpy(node_labels), num_classes=10).float()
            g.ndata["x"] = x

            graph_label = int(np.load(glabel_path, allow_pickle=True))

            graphs.append(g)
            labels.append(graph_label)

    return graphs, labels


def _pd_file_for_graph_index(idx: int, window_size: int = 5, max_end: int = 98) -> str:
    """
    Map graph index -> PD window file name.
    Rule from collaborator note: first 5 graphs share one window (0_4).
    Tail is clipped to available max_end (e.g., graph_99 -> 94_98 when 95_99 absent).
    """
    if idx < window_size:
        s, e = 0, window_size - 1
    else:
        e = min(idx, max_end)
        s = max(0, e - (window_size - 1))
    return f"{s}_{e}_pd.npy"


def _load_pd_pair(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    if not isinstance(arr, np.ndarray) or arr.dtype != object or len(arr) < 2:
        raise ValueError(f"Invalid PD format at: {path}")

    h0 = np.asarray(arr[0], dtype=np.float32)
    h1 = np.asarray(arr[1], dtype=np.float32)

    h0 = h0.reshape(-1, 2) if h0.size > 0 else np.zeros((0, 2), dtype=np.float32)
    h1 = h1.reshape(-1, 2) if h1.size > 0 else np.zeros((0, 2), dtype=np.float32)
    return h0, h1


def _pd_hist2d(points: np.ndarray, bins: int, vmin: float, vmax: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((bins, bins), dtype=np.float32)

    x = np.clip(points[:, 0], vmin, vmax)
    y = np.clip(points[:, 1], vmin, vmax)
    h, _, _ = np.histogram2d(x, y, bins=bins, range=[[vmin, vmax], [vmin, vmax]])
    h = h.astype(np.float32)
    s = h.sum()
    if s > 0:
        h = h / s
    return h


def load_topology_images(topology_dir: str, bins: int = 32, window_size: int = 5) -> torch.Tensor:
    """
    Build topology images aligned with graph order used in load_dropbox_graphs:
      [G1_0..99, G2_0..99].

    Each PD file stores object[2] = [H0_points, H1_points].
    We convert them into 2-channel persistence hist images, then feed to CNN.
    """
    pd_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    all_vals: List[np.ndarray] = []

    prefix_dir = {"G1": "graph_1", "G2": "graph_2"}

    for prefix in ["G1", "G2"]:
        for idx in range(100):
            pd_name = _pd_file_for_graph_index(idx, window_size=window_size, max_end=98)
            pd_path = os.path.join(topology_dir, prefix_dir[prefix], pd_name)
            if not os.path.exists(pd_path):
                raise FileNotFoundError(f"Missing topology PD file: {pd_path}")

            h0, h1 = _load_pd_pair(pd_path)
            pd_pairs.append((h0, h1))
            if h0.size > 0:
                all_vals.append(h0.reshape(-1))
            if h1.size > 0:
                all_vals.append(h1.reshape(-1))

    if len(all_vals) > 0:
        vals = np.concatenate(all_vals)
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    imgs = []
    for h0, h1 in pd_pairs:
        img0 = _pd_hist2d(h0, bins=bins, vmin=vmin, vmax=vmax)
        img1 = _pd_hist2d(h1, bins=bins, vmin=vmin, vmax=vmax)
        imgs.append(np.stack([img0, img1], axis=0))  # [2, bins, bins]

    return torch.from_numpy(np.stack(imgs, axis=0)).float()  # [N, 2, bins, bins]


def stratified_split(labels: List[int], train_ratio=0.7, val_ratio=0.15, seed=42) -> SplitIndices:
    rng = random.Random(seed)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val: n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def index_block_split(labels: List[int], train_ratio=0.7, val_ratio=0.15) -> SplitIndices:
    """
    Strict split without shuffling within each class (index-order split).
    Useful as a stronger anti-leakage sanity check when indices may reflect
    generation order/time.
    """
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for _, idxs in by_class.items():
        idxs = sorted(idxs)
        n = len(idxs)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val: n_train + n_val + n_test])

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def collate_fn(batch):
    if len(batch[0]) == 3:
        graphs, labels, topos = map(list, zip(*batch))
        bg = dgl.batch(graphs)
        y = torch.stack(labels)
        topo = torch.stack(topos)
        return bg, y, topo

    graphs, labels = map(list, zip(*batch))
    bg = dgl.batch(graphs)
    y = torch.stack(labels)
    return bg, y


def evaluate(model, loader, device, use_topology_feature: bool = False):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        if use_topology_feature:
            for bg, y, topo in loader:
                bg = bg.to(device)
                y = y.to(device)
                topo = topo.to(device)
                logits = model(bg, topo)
                pred = logits.argmax(dim=1)
                ys.extend(y.cpu().tolist())
                ps.extend(pred.cpu().tolist())
        else:
            for bg, y in loader:
                bg = bg.to(device)
                y = y.to(device)
                logits = model(bg)
                pred = logits.argmax(dim=1)
                ys.extend(y.cpu().tolist())
                ps.extend(pred.cpu().tolist())

    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    return acc, f1


def train_one_run(args, run_seed, graphs, labels, device, topo_images: Optional[torch.Tensor] = None):
    if args.split_mode == "index_block":
        split = index_block_split(labels, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    else:
        split = stratified_split(labels, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=run_seed)

    train_ds = GraphDataset(graphs, labels, split.train, topo_images=topo_images)
    val_ds = GraphDataset(graphs, labels, split.val, topo_images=topo_images)
    test_ds = GraphDataset(graphs, labels, split.test, topo_images=topo_images)

    if args.label_shuffle_train:
        g = torch.Generator()
        g.manual_seed(run_seed)
        perm = torch.randperm(len(train_ds.labels), generator=g)
        train_ds.labels = train_ds.labels[perm]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GraphClassifier(
        input_dim=10,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        mode=args.mode,
        num_experts=args.num_experts,
        moe_n_heads=args.moe_n_heads,
        moe_top_k=args.moe_top_k,
        dropout=args.dropout,
        use_topology_feature=args.use_topology_feature,
        topo_alpha=args.topo_alpha,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        if args.use_topology_feature:
            for bg, y, topo in train_loader:
                bg = bg.to(device)
                y = y.to(device)
                topo = topo.to(device)
                logits = model(bg, topo)
                loss = criterion(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            for bg, y in train_loader:
                bg = bg.to(device)
                y = y.to(device)
                logits = model(bg)
                loss = criterion(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        val_acc, val_f1 = evaluate(model, val_loader, device, use_topology_feature=args.use_topology_feature)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    train_acc, train_f1 = evaluate(model, train_loader, device, use_topology_feature=args.use_topology_feature)
    val_acc, val_f1 = evaluate(model, val_loader, device, use_topology_feature=args.use_topology_feature)
    test_acc, test_f1 = evaluate(model, test_loader, device, use_topology_feature=args.use_topology_feature)

    return {
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if (args.gpu >= 0 and torch.cuda.is_available()) else "cpu")

    graphs, labels = load_dropbox_graphs(args.data_dir)

    topo_images = None
    if args.use_topology_feature:
        topo_images = load_topology_images(
            topology_dir=args.topology_dir,
            bins=args.topo_bins,
            window_size=args.topo_window,
        )

    runs = []
    for r in range(args.runs):
        run_seed = args.seed + r
        result = train_one_run(args, run_seed, graphs, labels, device, topo_images=topo_images)
        runs.append(result)
        print(
            f"run={r} seed={run_seed} "
            f"test_acc={result['test_acc']:.4f} test_f1={result['test_f1']:.4f} "
            f"(train/val/test={result['n_train']}/{result['n_val']}/{result['n_test']})"
        )

    mean_test_acc = float(np.mean([x["test_acc"] for x in runs]))
    std_test_acc = float(np.std([x["test_acc"] for x in runs]))
    mean_test_f1 = float(np.mean([x["test_f1"] for x in runs]))
    std_test_f1 = float(np.std([x["test_f1"] for x in runs]))

    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Use topology feature: {args.use_topology_feature}")
    if args.use_topology_feature:
        print(f"Topology dir: {args.topology_dir}")
        print(f"Topology bins: {args.topo_bins} | Topology window: {args.topo_window} | Topology alpha: {args.topo_alpha}")
    print(f"Split mode: {args.split_mode}")
    print(f"Label shuffle(train): {args.label_shuffle_train}")
    print(f"Runs: {args.runs}")
    print(f"Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
    print(f"Test Macro-F1: {mean_test_f1:.4f} ± {std_test_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph classification on Dropbox G1/G2 with GNN/MoE variants")
    parser.add_argument("--data_dir", type=str, default="../dropbox_import_20260219")
    parser.add_argument("--mode", type=str, default="parallel", choices=["gnn", "serial", "parallel"])
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--hidden_dim", type=int, default=48)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--moe_n_heads", type=int, default=4)
    parser.add_argument("--moe_top_k", type=int, default=2)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "index_block"],
                        help="random stratified split or strict index-order split per class")
    parser.add_argument("--label_shuffle_train", action="store_true",
                        help="shuffle train labels as sanity-check (should drop to near chance)")

    parser.add_argument("--use_topology_feature", action="store_true",
                        help="enable topology branch: PD(H0/H1)->CNN->ExpandMLP, then fuse with graph embedding")
    parser.add_argument("--topology_dir", type=str, default="../topology_import_20260220/output",
                        help="directory containing graph_1/graph_2 with *_pd.npy files")
    parser.add_argument("--topo_bins", type=int, default=32,
                        help="histogram bins per axis for persistence image")
    parser.add_argument("--topo_window", type=int, default=5,
                        help="window size used for PD file mapping (default 5)")
    parser.add_argument("--topo_alpha", type=float, default=1.0,
                        help="scaling weight for topology branch embedding before fusion")

    args = parser.parse_args()
    main(args)

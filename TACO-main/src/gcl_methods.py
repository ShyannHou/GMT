import argparse
from dgl.data import register_data_args
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

import numpy as np

import quadprog
import torch.optim as optim
import copy
import random
import collections


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel_bandwidth: float = 1.0) -> torch.Tensor:
    """Gaussian-kernel Maximum Mean Discrepancy (MMD).

    Adapted from DecAlign (ICLR'26 under review) compute_mmd.
    x, y: [N, D]
    """
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    K_xx = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * kernel_bandwidth))
    K_yy = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * kernel_bandwidth))
    K_xy = torch.exp(- (rx.t() + ry - 2 * xy) / (2 * kernel_bandwidth))
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


class DYGRA_reservior(torch.nn.Module):
    def __init__(self,
                 model, opt, num_class, buffer_size,
                 args):
        super(DYGRA_reservior, self).__init__()

        self.net = model

        self.opt = opt

        self.current_task = 0
        self.num_samples = 0
        self.buffer_size = buffer_size
        self.er_buffer = []

        # Optional (DYGRA-only) modality alignment settings
        self.args = args

    def update_er_buffer(self, g):
        train_nodes = g.ndata['train_mask'].nonzero()
        for node in train_nodes:
            node_feature, node_label = g.ndata['x'][node], g.ndata['y'][node].item()
            self.num_samples += 1
            if len(self.er_buffer) < self.buffer_size:
                self.er_buffer.append(g.ndata['node_idxs'][node])
            else:
                rand_idx = random.randint(0, self.num_samples-1)
                if rand_idx < self.buffer_size:
                    self.er_buffer[rand_idx] = g.ndata['node_idxs'][node]
        return self.er_buffer

    def forward(self, g, features):

        output = self.net(g, features)
        return output

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['valid_mask']
        self.net.train()
        self.net.zero_grad()

        lam = float(getattr(self.args, 'dygra_mmd_lambda', 0.0))
        bw = float(getattr(self.args, 'dygra_mmd_bandwidth', 1.0))
        sample_n = int(getattr(self.args, 'dygra_mmd_sample', 256))

        h = h_moe = topo_h = None
        if lam > 0:
            # Only works when the model supports return_parts (e.g., ParallelGNNMoE).
            try:
                output, h, h_moe, topo_h = self.net(g, features, return_parts=True)
            except TypeError:
                output = self.net(g, features)
        else:
            output = self.net(g, features)

        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])

        # DYGRA-only modality alignment (MMD) among {GNN hidden, MoE, topology/TDA}.
        if lam > 0 and (h is not None) and (h_moe is not None) and (topo_h is not None):
            idx = train_mask.nonzero(as_tuple=False).view(-1)
            if sample_n > 0 and idx.numel() > sample_n:
                perm = torch.randperm(idx.numel(), device=idx.device)[:sample_n]
                idx = idx[perm]

            x = h[idx].float()
            y = h_moe[idx].float()
            z = topo_h[idx].float()
            L_mmd = compute_mmd(x, y, kernel_bandwidth=bw) + compute_mmd(x, z, kernel_bandwidth=bw) + compute_mmd(y, z, kernel_bandwidth=bw)
            loss = loss + lam * L_mmd

        loss.backward()
        self.opt.step()


class DYGRA_meanfeature(torch.nn.Module):
    def __init__(self,
                 model, opt, num_class, buffer_size,
                 args):
        super(DYGRA_meanfeature, self).__init__()

        self.net = model

        self.opt = opt

        self.current_task = 0
        self.num_samples = 0
        self.num_class = num_class
        self.buffer_size_per_class = buffer_size // num_class
        self.moving_avg = [-1 for i in range(num_class)]
        self.buffer_size = buffer_size
        self.er_buffer = [[] for _ in range(num_class)]
        self.appeared_samples = [0 for i in range(num_class)]

    def update_er_buffer(self, g, t):
        g = copy.deepcopy(g)
        g.add_edges(g.nodes(), g.nodes())
        self.net.return_hidden = True

        hidden_features = self.net(g, g.ndata['x'])
        train_nodes = g.ndata['train_mask'].nonzero()
        for node in train_nodes:
            node_feature, node_label = g.ndata['x'][node], g.ndata['y'][node].item()
            node_hidden_feature = hidden_features[node]
            n = self.appeared_samples[node_label]
            if n == 0:
                self.moving_avg[node_label] = node_hidden_feature
            else:
                self.moving_avg[node_label] = self.moving_avg[node_label] / (n + 1) * n + node_hidden_feature / (n + 1)
            d = torch.cdist(self.moving_avg[node_label], node_hidden_feature)

            if len(self.er_buffer[node_label]) < self.buffer_size_per_class:
                self.er_buffer[node_label].append((node_feature, d, node))
            else:
                max_d = max(self.er_buffer[node_label], key=lambda x: x[1])

                max_d_idx = [x[1] for x in self.er_buffer[node_label]].index(max_d[1])
                if d < max_d[1]:
                    self.er_buffer[node_label][max_d_idx] = (node_feature, d, node)
            self.appeared_samples[node_label] += 1
        self.net.return_hidden = False

        er_nodes = []
        for i in range(len(self.er_buffer)):
            er_nodes.extend((x[2] for x in self.er_buffer[i]))
        return er_nodes

    def forward(self, g, features):
        output = self.net(g, features)
        return output

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']
        valid_mask = g.ndata['valid_mask']
        self.net.train()
        self.net.zero_grad()

        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])
        loss.backward()
        self.opt.step()


class DYGRA_ringbuffer(nn.Module):
    def __init__(self,
                 model,
                 opt, num_class, buffer_size,
                 args, device=torch.device('cuda:0')):
        super(DYGRA_ringbuffer, self).__init__()
        self.net = model
        self.opt = opt
        self.buffer_size_per_class = buffer_size // num_class
        self.er_buffer = [collections.deque([], self.buffer_size_per_class) for _ in range(num_class)]
        self.num_samples = 0
        self.num_class = num_class

        self.device = device

    def forward(self, g, features):
        g = copy.deepcopy(g)
        g.add_edges(g.nodes(), g.nodes())
        output = self.net(g, features)
        return output

    def update_er_buffer(self, g, t):

        train_nodes = g.ndata['train_mask'].nonzero()
        for node in train_nodes:
            node_feature, node_label = g.ndata['x'][node], g.ndata['y'][node].item()
            self.er_buffer[node_label].appendleft((node_feature, node))
        er_nodes = []
        for i in range(len(self.er_buffer)):
            er_nodes.extend([x[1] for x in self.er_buffer[i]])
        return er_nodes

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = g.ndata['y']
        train_mask = g.ndata['train_mask']

        self.net.train()
        self.net.zero_grad()
        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func((output[train_mask]), labels[train_mask])
        if t > 0:
            g_rp = dgl.DGLGraph().to(self.device)

            for node_label in range(self.num_class):
                if len(self.er_buffer[node_label]) > 0:
                    x = torch.cat(list([z[0] for z in self.er_buffer[node_label]]))
                    label = torch.tensor([node_label for _ in range(len(self.er_buffer[node_label]))]).to(self.device)
                    g_rp.add_nodes(len(self.er_buffer[node_label]), {'x': x, 'y': label})

            g_rp.add_edges(g_rp.nodes(), g_rp.nodes())
            output = self.net(g_rp, g_rp.ndata['x'])
            output = F.log_softmax(output, 1)

            loss = loss * 0.5 + (1 - 0.5) * loss_func(output, g_rp.ndata['y'])
        loss.backward()
        self.opt.step()


# -------------------------
# Baselines inside pipeline
# -------------------------

class FINETUNE(torch.nn.Module):
    """Finetune baseline inside the same training/coarsening pipeline.

    - No replay buffer
    - Train only on current task training nodes.

    This makes it comparable to DYGRA under the same combine_graph + coarsening flow.
    """

    def __init__(self, model, opt, num_class, buffer_size, args):
        super().__init__()
        self.net = model
        self.opt = opt

    def update_er_buffer(self, g):
        return []

    def forward(self, g, features):
        return self.net(g, features)

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']

        self.net.train()
        self.net.zero_grad()
        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func(output[train_mask], labels[train_mask])
        loss.backward()
        self.opt.step()


class SIMPLE_REG(torch.nn.Module):
    """Simple L2 regularization to previous task parameters, inside pipeline."""

    def __init__(self, model, opt, num_class, buffer_size, args):
        super().__init__()
        self.net = model
        self.opt = opt
        self.args = args
        self.prev_params = None
        self.prev_task = None

    def update_er_buffer(self, g):
        return []

    def forward(self, g, features):
        return self.net(g, features)

    def _maybe_snapshot_prev(self, t):
        if self.prev_task is None:
            self.prev_task = t
            return
        if t != self.prev_task:
            if t > 0:
                self.prev_params = {n: p.detach().clone() for n, p in self.net.named_parameters()}
            self.prev_task = t

    def observe(self, g_list, t, loss_func):
        self._maybe_snapshot_prev(t)

        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']

        self.net.train()
        self.net.zero_grad()
        output = self.net(g, features)
        output = F.log_softmax(output, 1)
        loss = loss_func(output[train_mask], labels[train_mask])

        lam = float(getattr(self.args, 'simple_reg_lambda', 0.0))
        if self.prev_params is not None and lam > 0:
            reg = 0.0
            for n, p in self.net.named_parameters():
                reg = reg + (p - self.prev_params[n].to(p.device)).pow(2).sum()
            loss = loss + lam * reg

        loss.backward()
        self.opt.step()


# -------------------------
# Additional NCGL baselines
#   - BARE: naive sequential
#   - ERGNN: experience replay (node buffer)
#   - GEM: gradient episodic memory (quadprog)
#   - TWP: topology-aware weight preservation (fisher + grad-norm)
#
# Implementations are adapted to TACO's full-graph + coarsening pipeline and
# follow the reference implementations in CGLB (NCGL/Baselines).
# -------------------------


class BARE(FINETUNE):
    """Alias of FINETUNE (naive sequential training) for naming compatibility."""


def _store_grad(pp, grads: torch.Tensor, grad_dims, tid: int):
    """Store current parameter grads into grads[:, tid]."""
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def _overwrite_grad(pp, newgrad: torch.Tensor, grad_dims):
    """Overwrite param.grad with newgrad vector."""
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def _project2cone2(gradient: torch.Tensor, memories: torch.Tensor, margin: float = 0.5, eps: float = 1e-3):
    """GEM projection via the dual QP (quadprog), adapted from CGLB."""
    # memories: [p, t]
    memories_np = memories.detach().cpu().t().double().numpy()
    gradient_np = gradient.detach().cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]

    P = memories_np @ memories_np.T
    P = 0.5 * (P + P.T) + np.eye(t) * eps
    q = (memories_np @ gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = v @ memories_np + gradient_np

    gradient.copy_(torch.tensor(x, dtype=gradient.dtype, device=gradient.device).view(-1, 1))


class GEM(torch.nn.Module):
    """Gradient Episodic Memory (GEM) baseline.

    Notes for this codebase:
      - We store a small memory set (features+labels) per task.
      - Old-task gradients are computed on the stored memory graphs (self-loops only).
      - Current-task gradient is projected (quadprog) to not interfere with past tasks.

    Reference: CGLB/NCGL/Baselines/gem_model.py (+ gem_utils.py).
    """

    def __init__(self, model, opt, num_class, buffer_size, args):
        super().__init__()
        self.net = model
        self.opt = opt
        self.args = args

        self.margin = float(getattr(args, 'gem_memory_strength', 0.5))
        self.n_memories = int(getattr(args, 'gem_n_memories', 100))

        self.observed_tasks = []
        self.current_task = -1

        self.memory_feats = []   # list[t] -> Tensor [M, D]
        self.memory_labels = []  # list[t] -> Tensor [M]

        self.grad_dims = [p.data.numel() for p in self.net.parameters()]
        self.grads = None  # allocated lazily: [sum_dims, num_tasks]

    def update_er_buffer(self, g):
        # GEM does not require replay_nodes for coarsening (kept empty for fairness).
        return []

    def forward(self, g, features):
        return self.net(g, features)

    def _ensure_grads(self, num_tasks: int, device: torch.device):
        if self.grads is None or self.grads.size(1) < num_tasks:
            self.grads = torch.zeros(sum(self.grad_dims), num_tasks, device=device)

    def _maybe_update_memory(self, g: dgl.DGLGraph, t: int):
        # Called once per task (first time we see t). Sample memory from current train set.
        train_nodes = g.ndata['train_mask'].nonzero(as_tuple=False).view(-1)
        if train_nodes.numel() == 0:
            feats = g.ndata['x'][:0]
            labs = torch.zeros((0,), device=g.device, dtype=torch.long)
        else:
            k = min(self.n_memories, train_nodes.numel())
            perm = torch.randperm(train_nodes.numel(), device=train_nodes.device)[:k]
            idx = train_nodes[perm]
            feats = g.ndata['x'][idx].detach().clone()
            y = g.ndata['y']
            labs = (torch.max(y, 1).indices if y.dim() == 2 else y).detach().clone()[idx]

        while len(self.memory_feats) <= t:
            self.memory_feats.append(None)
            self.memory_labels.append(None)
        self.memory_feats[t] = feats
        self.memory_labels[t] = labs

    def _loss_on_memory(self, feats: torch.Tensor, labs: torch.Tensor, loss_func):
        # Self-loop-only replay graph
        n = feats.size(0)
        g_rp = dgl.graph(([], []), num_nodes=n, device=feats.device)
        if n > 0:
            nodes = torch.arange(n, device=feats.device)
            g_rp.add_edges(nodes, nodes)
        logits = self.net(g_rp, feats)
        out = F.log_softmax(logits, 1)
        return loss_func(out, labs)

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']

        self.net.train()

        self._ensure_grads(num_tasks=len(g_list), device=features.device)

        # task boundary
        if t != self.current_task:
            self.observed_tasks.append(t)
            self.current_task = t
            self._maybe_update_memory(g, t)

        # 1) gradients on past task memories
        for old_t in self.observed_tasks[:-1]:
            feats_old = self.memory_feats[old_t]
            labs_old = self.memory_labels[old_t]
            if feats_old is None or feats_old.numel() == 0:
                continue
            self.net.zero_grad()
            loss_old = self._loss_on_memory(feats_old, labs_old, loss_func)
            loss_old.backward()
            _store_grad(self.net.parameters, self.grads, self.grad_dims, old_t)

        # 2) gradient on current task
        self.net.zero_grad()
        logits = self.net(g, features)
        out = F.log_softmax(logits, 1)
        loss = loss_func(out[train_mask], labels[train_mask])
        loss.backward()

        # 3) project if constraints violated
        if len(self.observed_tasks) > 1:
            _store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.tensor(self.observed_tasks[:-1], device=features.device, dtype=torch.long)
            dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
            if (dotp < 0).any():
                _project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), margin=self.margin)
                _overwrite_grad(self.net.parameters, self.grads[:, t], self.grad_dims)

        self.opt.step()


class ERGNN(torch.nn.Module):
    """Experience Replay GNN (ER-GNN) baseline.

    We implement a simplified ER-GNN variant suitable for TACO's pipeline:
      - Maintain a reservoir buffer of training nodes (for coarsening fidelity) via node_idxs.
      - Additionally store (feature,label) for replay training on a self-loop graph.

    Reference: CGLB/NCGL/Baselines/ergnn_model.py.
    """

    def __init__(self, model, opt, num_class, buffer_size, args):
        super().__init__()
        self.net = model
        self.opt = opt
        self.args = args

        self.buffer_size = int(getattr(args, 'ergnn_budget', buffer_size))
        self.num_samples = 0
        self.er_buffer = []  # stores original node_idxs (ints) for coarsening
        self.replay_feats = []
        self.replay_labels = []

    def update_er_buffer(self, g: dgl.DGLGraph):
        train_nodes = g.ndata['train_mask'].nonzero(as_tuple=False).view(-1)
        y = g.ndata['y']
        y_int = torch.max(y, 1).indices if y.dim() == 2 else y

        for node in train_nodes:
            node = int(node.item())
            node_idx = int(g.ndata['node_idxs'][node].item())
            feat = g.ndata['x'][node].detach().clone()
            lab = int(y_int[node].item())

            self.num_samples += 1
            if len(self.er_buffer) < self.buffer_size:
                self.er_buffer.append(node_idx)
                self.replay_feats.append(feat)
                self.replay_labels.append(lab)
            else:
                rand_idx = random.randint(0, self.num_samples - 1)
                if rand_idx < self.buffer_size:
                    self.er_buffer[rand_idx] = node_idx
                    self.replay_feats[rand_idx] = feat
                    self.replay_labels[rand_idx] = lab

        return self.er_buffer

    def forward(self, g, features):
        return self.net(g, features)

    def observe(self, g_list, t, loss_func):
        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']

        self.net.train()
        self.net.zero_grad()

        logits = self.net(g, features)
        out = F.log_softmax(logits, 1)
        loss_cur = loss_func(out[train_mask], labels[train_mask])

        # Replay loss (self-loop graph), weighted as in ER-GNN (beta = B/(B+N)).
        if t > 0 and len(self.replay_feats) > 0:
            feats_rp = torch.stack([f.squeeze(0) if f.dim() == 2 else f for f in self.replay_feats], dim=0).to(features.device)
            labs_rp = torch.tensor(self.replay_labels, device=features.device, dtype=torch.long)
            n = feats_rp.size(0)
            g_rp = dgl.graph(([], []), num_nodes=n, device=features.device)
            if n > 0:
                nodes = torch.arange(n, device=features.device)
                g_rp.add_edges(nodes, nodes)
            logits_rp = self.net(g_rp, feats_rp)
            out_rp = F.log_softmax(logits_rp, 1)
            loss_rp = loss_func(out_rp, labs_rp)

            n_nodes = int(train_mask.sum().item())
            beta = float(n) / float(max(n + n_nodes, 1))
            loss = beta * loss_cur + (1.0 - beta) * loss_rp
        else:
            loss = loss_cur

        loss.backward()
        self.opt.step()


class TWP(torch.nn.Module):
    """Topology-aware Weight Preservation (TWP) baseline.

    We adapt the CGLB implementation to this codebase:
      - fisher_loss: per-parameter squared gradients of task loss at the end of each task.
      - fisher_att: squared gradients of an "attention proxy" (here: MoE gate logits if present; otherwise logits norm).
      - Regularize parameters to stay near previous optima.

    Reference: CGLB/NCGL/Baselines/twp_model.py.
    """

    def __init__(self, model, opt, num_class, buffer_size, args):
        super().__init__()
        self.net = model
        self.opt = opt
        self.args = args

        self.lambda_l = float(getattr(args, 'twp_lambda_l', 10000.0))
        self.lambda_t = float(getattr(args, 'twp_lambda_t', 10000.0))
        self.beta = float(getattr(args, 'twp_beta', 0.01))

        self.epochs = 0
        self.fisher_loss = {}
        self.fisher_att = {}
        self.optpar = {}

    def update_er_buffer(self, g):
        return []

    def forward(self, g, features):
        return self.net(g, features)

    def _attention_proxy(self, logits: torch.Tensor):
        # Prefer MoE gate logits if available (connected to graph).
        moe = getattr(self.net, 'moe', None)
        if moe is not None and getattr(moe, 'last_gate_logits', None) is not None:
            return moe.last_gate_logits
        return logits

    def observe(self, g_list, t, loss_func):
        self.epochs += 1
        last_epoch = self.epochs % int(getattr(self.args, 'num_epochs', 50))

        g = copy.deepcopy(g_list[t])
        g.add_edges(g.nodes(), g.nodes())
        features = g.ndata['x']
        labels = torch.max(g.ndata['y'], 1).indices
        train_mask = g.ndata['train_mask']

        self.net.train()

        # Train step with TWP regularization
        self.net.zero_grad()
        logits = self.net(g, features)
        out = F.log_softmax(logits, 1)
        loss = loss_func(out[train_mask], labels[train_mask])

        # First backward to get gradients for grad-norm (official-style)
        loss.backward(retain_graph=True)
        grad_norm = 0.0
        for p in self.net.parameters():
            if p.grad is None:
                continue
            grad_norm = grad_norm + torch.norm(p.grad.data.clone(), p=1)

        # Quadratic penalties from previous tasks
        for tt in range(t):
            if tt not in self.fisher_loss:
                continue
            for i, p in enumerate(self.net.parameters()):
                l = self.lambda_l * self.fisher_loss[tt][i] + self.lambda_t * self.fisher_att[tt][i]
                l = l.to(p.device) * (p - self.optpar[tt][i].to(p.device)).pow(2)
                loss = loss + l.sum()

        loss = loss + self.beta * grad_norm

        # Second backward accumulates the full objective gradients
        loss.backward()
        self.opt.step()

        # At task boundary (end of task training), estimate fisher terms
        if last_epoch == 0:
            self.net.zero_grad()
            logits2 = self.net(g, features)
            out2 = F.log_softmax(logits2, 1)
            loss2 = loss_func(out2[train_mask], labels[train_mask])
            loss2.backward(retain_graph=True)

            self.fisher_loss[t] = []
            self.fisher_att[t] = []
            self.optpar[t] = []
            for p in self.net.parameters():
                self.optpar[t].append(p.data.detach().clone())
                if p.grad is None:
                    self.fisher_loss[t].append(torch.zeros_like(p.data))
                else:
                    self.fisher_loss[t].append(p.grad.data.detach().clone().pow(2))

            # attention proxy fisher
            att = self._attention_proxy(logits2)
            eloss = torch.norm(att)
            self.net.zero_grad()
            eloss.backward()
            for p in self.net.parameters():
                if p.grad is None:
                    self.fisher_att[t].append(torch.zeros_like(p.data))
                else:
                    self.fisher_att[t].append(p.grad.data.detach().clone().pow(2))

import argparse
import os
from dgl.data import register_data_args
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import pickle
import random
import numpy as np

import csv
import copy
import os
import time
import glob

from models import *
from moe_expert import MoEGNN, ParallelGNNMoE
from pd_image_model import CNN, ExpandMLP
from utils import *
from gcl_methods import *


def _pd_file_for_slot(slot, window_size=5, max_end=98):
    """
    Window mapping used by collaborator's topology package:
      - first window: 0_4
      - then slide by 1: 1_5, 2_6, ...
    For early slots (<window_size), share 0_4.
    """
    if slot < window_size:
        s, e = 0, window_size - 1
    else:
        e = min(slot, max_end)
        s = max(0, e - (window_size - 1))
    return f"{s}_{e}_pd.npy"


def _load_pd_pair(path):
    arr = np.load(path, allow_pickle=True)
    if not isinstance(arr, np.ndarray) or arr.dtype != object or len(arr) < 2:
        raise ValueError(f"Invalid PD format: {path}")

    h0 = np.asarray(arr[0], dtype=np.float32)
    h1 = np.asarray(arr[1], dtype=np.float32)

    h0 = h0.reshape(-1, 2) if h0.size > 0 else np.zeros((0, 2), dtype=np.float32)
    h1 = h1.reshape(-1, 2) if h1.size > 0 else np.zeros((0, 2), dtype=np.float32)
    return h0, h1


def _pd_hist2d(points, bins, vmin, vmax):
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


def _collect_pd_range(pd_paths):
    vals = []
    for p in pd_paths:
        h0, h1 = _load_pd_pair(p)
        if h0.size > 0:
            vals.append(h0.reshape(-1))
        if h1.size > 0:
            vals.append(h1.reshape(-1))

    if len(vals) == 0:
        return 0.0, 1.0

    v = np.concatenate(vals)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def build_topology_slot_embeddings(num_task, hidden_dim, num_experts, device,
                                   topology_pd_dir, topology_group=1,
                                   topology_window=5, topology_bins=32,
                                   topology_max_end=98, topo_seed=42):
    """
    Build static per-slot topology embeddings using:
      PD(H0,H1) -> 2ch persistence image -> CNN -> ExpandMLP -> mean over n.

    topology_group:
      - 1 or 2: use graph_1 / graph_2 only
      - 0: average embeddings from both graph_1 and graph_2

    Output: dict slot -> topo embedding tensor [hidden_dim] (CPU tensor).
    """
    group_ids = [1, 2] if int(topology_group) == 0 else [int(topology_group)]
    group_dirs = []
    all_pd_paths = []

    for gid in group_ids:
        gdir = os.path.join(topology_pd_dir, f"graph_{gid}")
        if not os.path.isdir(gdir):
            raise FileNotFoundError(f"Topology group dir not found: {gdir}")
        pd_paths = sorted(glob.glob(os.path.join(gdir, "*_pd.npy")))
        if len(pd_paths) == 0:
            raise FileNotFoundError(f"No topology PD files found under: {gdir}")
        group_dirs.append(gdir)
        all_pd_paths.extend(pd_paths)

    vmin, vmax = _collect_pd_range(all_pd_paths)

    torch.manual_seed(topo_seed)
    cnn = CNN(input_channel=2, dim_out=hidden_dim).to(device)
    expand = ExpandMLP(d=hidden_dim, n=max(1, int(num_experts))).to(device)
    cnn.eval()
    expand.eval()

    slot_embed = {}
    with torch.no_grad():
        for slot in range(num_task):
            pd_name = _pd_file_for_slot(slot, window_size=topology_window, max_end=topology_max_end)
            slot_group_embeds = []

            for gdir in group_dirs:
                pd_path = os.path.join(gdir, pd_name)
                if not os.path.exists(pd_path):
                    raise FileNotFoundError(f"Missing topology file for slot {slot}: {pd_path}")

                h0, h1 = _load_pd_pair(pd_path)
                img0 = _pd_hist2d(h0, bins=topology_bins, vmin=vmin, vmax=vmax)
                img1 = _pd_hist2d(h1, bins=topology_bins, vmin=vmin, vmax=vmax)
                topo_img = np.stack([img0, img1], axis=0)  # [2, H, W]
                topo_img = torch.from_numpy(topo_img).unsqueeze(0).float().to(device)

                z = cnn(topo_img)               # [1, D]
                z_n = expand(z)                 # [1, n, D]
                z = z_n.mean(dim=1).squeeze(0)  # [D]
                slot_group_embeds.append(z)

            z_slot = torch.stack(slot_group_embeds, dim=0).mean(dim=0)
            slot_embed[slot] = z_slot.detach().cpu()

    return slot_embed


def run(args):
    use_gpu = (args.gpu == 0)
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    num_runs = args.num_runs
    datapath = f'../data/{args.Dataset}/'
    with open(datapath+'statistics', 'rb') as file:
        num_task, num_class = pickle.load(file)
    ave_test_acc, ave_test_bac, ave_test_f1 = 0.,0.,0.
    ave_test_acc_list = [0. for i in range(num_task)]
    ave_test_bac_list = [0. for i in range(num_task)]
    ave_test_f1_list = [0. for i in range(num_task)]

    ave_test_acc_raw = [[[0. for i in range(num_runs)] for i in range(num_task+1)] for i in range(num_task+1)]
    ave_test_bac_raw = [[[0. for i in range(num_runs)] for i in range(num_task+1)] for i in range(num_task+1)]
    ave_test_f1_raw = [[[0. for i in range(num_runs)] for i in range(num_task+1)] for i in range(num_task+1)]

    topology_slot_embed = None
    if args.use_topology_feature:
        topology_slot_embed = build_topology_slot_embeddings(
            num_task=num_task,
            hidden_dim=hidden_dim,
            num_experts=args.num_experts,
            device=device,
            topology_pd_dir=args.topology_pd_dir,
            topology_group=args.topology_group,
            topology_window=args.topology_window,
            topology_bins=args.topology_bins,
            topology_max_end=args.topology_max_end,
            topo_seed=args.topo_seed,
        )

    for run in range(num_runs):

        g_list = []
        with open(datapath+f'mask_seed_{run}', 'rb') as file:
            masks_supgraphs_list = pickle.load(file)
        for time_slot in range(num_task):
            with open(datapath+f'sub_graph_{time_slot}_by_edges', 'rb') as file:
                g = pickle.load(file)
            n_nodes = g.num_nodes()
            train_mask, valid_mask, test_mask = masks_supgraphs_list[time_slot]
            g.ndata['train_mask'] = train_mask
            g.ndata['valid_mask'] = valid_mask
            g.ndata['test_mask'] = test_mask

            if use_gpu:
                g = g.to(device)

            if topology_slot_embed is not None:
                topo_vec = topology_slot_embed[time_slot].to(device)
                g.ndata['topo_feat'] = topo_vec.unsqueeze(0).repeat(n_nodes, 1)

            g_list.append(g)
        input_dim = g.ndata['x'].size()[1]

        if args.method == "DYGRA":
            gcl_method = DYGRA_reservior
        elif args.method == "DYGRA_meanfeature":
            gcl_method = DYGRA_meanfeature
        elif args.method == "DYGRA_ringbuffer":
            gcl_method = DYGRA_ringbuffer
        elif args.method in {"FINETUNE", "SIMPLE_REG", "JOINT"}:
            gcl_method = None  # handled by baseline branch below
        else:
            raise ValueError(f"Unknown method: {args.method}")

        test_acc_list, test_bac_list, test_f1_list = [],[],[]
        num_test_list = []
        best_model_path = 'best_model_stat_dict'
        if args.gnn == "GAT":
            backbone = GAT(input_dim, hidden_dim, num_class, num_heads)
        elif args.gnn == "GCN":
            backbone = GCN(input_dim, hidden_dim, num_class)
        elif args.gnn == "GIN":
            backbone = GIN(input_dim, hidden_dim, num_class)

        if args.use_moe:
            if args.parallel_fusion:
                net = ParallelGNNMoE(
                    backbone,
                    input_dim,
                    hidden_dim,
                    num_class,
                    num_experts=args.num_experts,
                    n_heads=args.moe_n_heads,
                    top_k=args.moe_top_k,
                    use_topology_feature=args.use_topology_feature,
                    topo_alpha=args.topo_alpha,
                    topo_trainable_fusion=args.topo_trainable_fusion,
                )
            else:
                net = MoEGNN(
                    backbone,
                    hidden_dim,
                    num_class,
                    num_experts=args.num_experts,
                    n_heads=args.moe_n_heads,
                    top_k=args.moe_top_k,
                )
        else:
            net = backbone
        if use_gpu:
            net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_func = nn.CrossEntropyLoss()

        def _labels_from_graph(_g):
            y = _g.ndata['y']
            if len(y.size()) == 2:
                return torch.max(y, 1).indices
            return y

        if args.method in {"FINETUNE", "SIMPLE_REG", "JOINT"}:
            # -----------------
            # Baseline methods
            # -----------------
            prev_params = None

            if args.method == "JOINT":
                # Oracle-ish joint training: in each epoch, iterate through all tasks
                # and take one optimizer step per task (roughly comparable total steps
                # to FINETUNE which trains num_epochs per task).
                best_bac = -1.0
                for epoch in range(args.num_epochs):
                    net.train()
                    for tg in g_list:
                        g_train = copy.deepcopy(tg)
                        g_train.add_edges(g_train.nodes(), g_train.nodes())
                        feats = g_train.ndata['x']
                        labs = _labels_from_graph(g_train)
                        m = g_train.ndata['train_mask']

                        optimizer.zero_grad()
                        logits = net(g_train, feats)
                        loss = loss_func(logits[m], labs[m])
                        loss.backward()
                        optimizer.step()

                    # validation score: mean BAC over tasks
                    net.eval()
                    with torch.no_grad():
                        avg_valid_bac = 0.0
                        for vg in g_list:
                            g_val = copy.deepcopy(vg)
                            feats_v = g_val.ndata['x']
                            labs_v = _labels_from_graph(g_val)
                            m_v = g_val.ndata['valid_mask']
                            valid_bac, _, _ = evaluate(net, g_val, feats_v, labs_v, m_v)
                            avg_valid_bac += valid_bac / num_task
                    if avg_valid_bac > best_bac:
                        best_bac = avg_valid_bac
                        torch.save(net.state_dict(), best_model_path)

                net.load_state_dict(torch.load(best_model_path))

                for train_slot in range(num_task):
                    avg_test_acc, avg_test_bac, avg_test_f1 = 0.0, 0.0, 0.0
                    for test_slot in range(num_task):
                        g = g_list[test_slot]
                        feats = g.ndata['x']
                        labs = _labels_from_graph(g)
                        m = g.ndata['test_mask']
                        test_bac, test_f1, test_acc = evaluate(net, copy.deepcopy(g), feats, labs, m)
                        avg_test_acc += test_acc / num_task
                        avg_test_bac += test_bac / num_task
                        avg_test_f1 += test_f1 / num_task
                        ave_test_acc_raw[train_slot][test_slot][run] = test_acc
                        ave_test_bac_raw[train_slot][test_slot][run] = test_bac
                        ave_test_f1_raw[train_slot][test_slot][run] = test_f1
                    ave_test_acc_raw[train_slot][-1][run] = avg_test_acc
                    ave_test_bac_raw[train_slot][-1][run] = avg_test_bac
                    ave_test_f1_raw[train_slot][-1][run] = avg_test_f1

            else:
                for train_slot in range(num_task):
                    print('train_slot:', train_slot)

                    if args.method == "SIMPLE_REG" and train_slot > 0:
                        prev_params = {n: p.detach().clone() for n, p in net.named_parameters()}
                    else:
                        prev_params = None

                    best_bac = -1.0
                    for epoch in range(args.num_epochs):
                        net.train()
                        g_train = copy.deepcopy(g_list[train_slot])
                        g_train.add_edges(g_train.nodes(), g_train.nodes())
                        feats = g_train.ndata['x']
                        labs = _labels_from_graph(g_train)
                        m_train = g_train.ndata['train_mask']

                        optimizer.zero_grad()
                        logits = net(g_train, feats)
                        loss = loss_func(logits[m_train], labs[m_train])

                        if args.method == "SIMPLE_REG" and prev_params is not None and args.simple_reg_lambda > 0:
                            reg = 0.0
                            for n, p in net.named_parameters():
                                reg = reg + (p - prev_params[n].to(p.device)).pow(2).sum()
                            loss = loss + args.simple_reg_lambda * reg

                        loss.backward()
                        optimizer.step()

                        # validation on current task
                        net.eval()
                        with torch.no_grad():
                            g_val = copy.deepcopy(g_list[train_slot])
                            feats_v = g_val.ndata['x']
                            labs_v = _labels_from_graph(g_val)
                            m_val = g_val.ndata['valid_mask']
                            valid_bac, _, _ = evaluate(net, g_val, feats_v, labs_v, m_val)
                        if valid_bac > best_bac:
                            best_bac = valid_bac
                            torch.save(net.state_dict(), best_model_path)

                    net.load_state_dict(torch.load(best_model_path))

                    # test on all tasks
                    avg_test_acc, avg_test_bac, avg_test_f1 = 0.0, 0.0, 0.0
                    for test_slot in range(num_task):
                        g = g_list[test_slot]
                        feats = g.ndata['x']
                        labs = _labels_from_graph(g)
                        m = g.ndata['test_mask']
                        test_bac, test_f1, test_acc = evaluate(net, copy.deepcopy(g), feats, labs, m)
                        avg_test_acc += test_acc / num_task
                        avg_test_bac += test_bac / num_task
                        avg_test_f1 += test_f1 / num_task
                        ave_test_acc_raw[train_slot][test_slot][run] = test_acc
                        ave_test_bac_raw[train_slot][test_slot][run] = test_bac
                        ave_test_f1_raw[train_slot][test_slot][run] = test_f1
                    ave_test_acc_raw[train_slot][-1][run] = avg_test_acc
                    ave_test_bac_raw[train_slot][-1][run] = avg_test_bac
                    ave_test_f1_raw[train_slot][-1][run] = avg_test_f1
                    print(avg_test_f1, avg_test_bac)

        else:
            buffer_size = args.buffer_size
            gcl = gcl_method(net, optimizer, num_class, buffer_size, args)
            combined_g_list = []
            for train_slot in range(num_task):
                print ('train_slot:', train_slot)
                g = g_list[train_slot]
                if args.nfp:
                    er_buffer = gcl.update_er_buffer(g)
                else:
                    er_buffer = []
                if train_slot == 0:
                    combined_g, c2n, n2c = combine_graph(g, device=device)
                else:
                    combined_g, c2n, n2c = combine_graph(g, coarsened_g, C, c2n, n2c, device=device)

                replay_nodes = n2c[torch.tensor(er_buffer)]
                combined_g_list.append(combined_g)

                features = combined_g.ndata['x']
                labels = torch.max(combined_g.ndata['y'],1).indices
                train_mask = combined_g.ndata['train_mask']
                valid_mask = combined_g.ndata['valid_mask']

                all_logits=[]
                best_bac = 0

                for epoch in range(args.num_epochs):
                    gcl.observe(combined_g_list, train_slot, loss_func)
                    valid_bac, valid_f1, valid_acc = evaluate(gcl, copy.deepcopy(combined_g), features, labels, valid_mask)
                    if valid_bac > best_bac:
                        torch.save(net.state_dict(), best_model_path)

                gcl.net.load_state_dict(torch.load(best_model_path))

                gcl.net.return_hidden = True
                combined_g_copy = copy.deepcopy(combined_g)
                combined_g_copy.add_edges(combined_g_copy.nodes(), combined_g_copy.nodes())
                node_hidden_features = gcl.net(combined_g_copy, combined_g_copy.ndata['x']).detach()
                gcl.net.return_hidden = False
                coarsened_g, C, c2n, n2c = graph_coarsening(
                    combined_g,
                    node_hidden_features,
                    c2n,
                    n2c,
                    0.6,
                    args.reduction_rate,
                    replay_nodes,
                    device=device,
                    target_nodes=args.target_nodes,
                )

                avg_test_acc, avg_test_bac, avg_test_f1 = 0., 0., 0.
                for test_slot in range(num_task):

                    g = g_list[test_slot]
                    features = g.ndata['x']
                    if test_slot <= train_slot:
                        labels = torch.max(g.ndata['y'],1).indices
                    else:
                        labels = g.ndata['y']
                    test_mask = g.ndata['test_mask']
                    test_bac, test_f1, test_acc = evaluate(gcl, copy.deepcopy(g), features, labels, test_mask)
                    avg_test_acc += test_acc/num_task
                    avg_test_bac += test_bac/num_task
                    avg_test_f1 += test_f1/num_task
                    ave_test_acc_raw[train_slot][test_slot][run] = test_acc
                    ave_test_bac_raw[train_slot][test_slot][run] = test_bac
                    ave_test_f1_raw[train_slot][test_slot][run] = test_f1
                ave_test_acc_raw[train_slot][-1][run] = avg_test_acc
                ave_test_bac_raw[train_slot][-1][run] = avg_test_bac
                ave_test_f1_raw[train_slot][-1][run] = avg_test_f1
                print (avg_test_f1, avg_test_bac)

        for i in range(num_task):
            ave_test_acc_raw[-1][i][run] = max([ave_test_acc_raw[j][i][run] for j in range(num_task)]) - ave_test_acc_raw[-2][i][run]
            ave_test_bac_raw[-1][i][run] = max([ave_test_bac_raw[j][i][run] for j in range(num_task)]) - ave_test_bac_raw[-2][i][run]
            ave_test_f1_raw[-1][i][run] = max([ave_test_f1_raw[j][i][run] for j in range(num_task)]) - ave_test_f1_raw[-2][i][run]

        ave_test_acc_raw[-1][-1][run] = sum([ave_test_acc_raw[-1][i][run] for i in range(num_task)])/num_task
        ave_test_bac_raw[-1][-1][run] = sum([ave_test_bac_raw[-1][i][run] for i in range(num_task)])/num_task
        ave_test_f1_raw[-1][-1][run] = sum([ave_test_f1_raw[-1][i][run] for i in range(num_task)])/num_task

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_each_run = (ave_test_acc_raw[-2][-1], ave_test_bac_raw[-2][-1], ave_test_f1_raw[-2][-1])
    pickle.dump(results_each_run, open(f'results/{args.method}_{args.gnn}_{args.Dataset}_reduction_{args.reduction_rate}', 'wb'))

    result_path = f'../results/{args.Dataset}/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    results_each_run = (ave_test_acc_raw[-2][-1], ave_test_bac_raw[-2][-1], ave_test_f1_raw[-2][-1])
    pickle.dump(results_each_run, open(f'results/{args.method}_{args.gnn}_{args.Dataset}_reduction_{args.reduction_rate}', 'wb'))

    f = open(result_path+f'{args.method}_{args.gnn}_{args.Dataset}_reduction_{args.reduction_rate}.csv', 'w')
    record_results(f, ave_test_acc_raw, ave_test_bac_raw, ave_test_f1_raw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCL')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Dataset", type=str, default='kindle',
                        help=False)
    parser.add_argument("--num_runs", type=int, default=10,
                        help="number of runs, default = 10")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="number of training epochs, default = 50")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--gnn', type=str, default='GCN',
                        help="basemodel")
    parser.add_argument('--method', type=str, default="DYGRA",
                        help="GCL method. Options: DYGRA, DYGRA_meanfeature, DYGRA_ringbuffer, FINETUNE, SIMPLE_REG, JOINT")
    parser.add_argument('--simple_reg_lambda', type=float, default=1e-2,
                        help="lambda for SIMPLE_REG baseline: L_task + lambda * ||theta-theta_prev||^2")
    parser.add_argument("--reduction_rate", type=float, default=0.5,
                    help="reduction rate")
    parser.add_argument("--target_nodes", type=int, default=None,
                    help="fixed target number of nodes after coarsening (overrides reduction_rate if set)")
    parser.add_argument("--nfp", type=bool, default=True,
                    help="node fidality preservation")

    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=48,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--buffer_size', type=int, default=200)

    parser.add_argument("--use_moe", type=bool, default=False,
                        help="whether to use MoE expert layer (strict Eq.10 via AttentionLayer)")
    parser.add_argument("--num_experts", type=int, default=4,
                        help="number of experts in MoE layer")
    parser.add_argument("--moe_n_heads", type=int, default=4,
                        help="number of attention heads per expert")
    parser.add_argument("--moe_top_k", type=int, default=2,
                        help="number of experts activated per node")
    parser.add_argument("--parallel_fusion", type=bool, default=False,
                        help="if True, run GNN and MoE in parallel, then fuse and use MLP")
    parser.add_argument("--use_topology_feature", action="store_true",
                        help="safe hook for topology feature residual in parallel model; no effect unless g.ndata['topo_feat'] is present")
    parser.add_argument("--topo_alpha", type=float, default=0.1,
                        help="weight for topology residual when --use_topology_feature is enabled")
    parser.add_argument("--topo_trainable_fusion", action="store_true",
                        help="learn topology fusion weight/gate instead of fixed topo_alpha")
    parser.add_argument("--topology_pd_dir", type=str, default="../topology_import_20260220/output",
                        help="topology PD root dir containing graph_1/graph_2")
    parser.add_argument("--topology_group", type=int, default=1,
                        help="which topology group folder to use (1 or 2)")
    parser.add_argument("--topology_window", type=int, default=5,
                        help="window size used by PD file naming")
    parser.add_argument("--topology_bins", type=int, default=32,
                        help="bins for persistence image conversion")
    parser.add_argument("--topology_max_end", type=int, default=98,
                        help="maximum PD window end index available")
    parser.add_argument("--topo_seed", type=int, default=42,
                        help="seed for static topology encoder initialization")

    args = parser.parse_args()

    run(args)

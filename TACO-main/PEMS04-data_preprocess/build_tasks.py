
import numpy as np
import csv
import torch
import dgl
import pickle
import os
from collections import defaultdict

NUM_TASKS = 10
NUM_CLASSES = 3
NUM_SENSORS = 307
TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.3, 0.2, 0.5

PEMS04_DIR = '../PEMS04/'
OUTPUT_DIR = '../data/PEMS04/'
NPZ_PATH = os.path.join(PEMS04_DIR, 'pems04.npz')
DISTANCE_PATH = os.path.join(PEMS04_DIR, 'distance.csv')

def load_pems04_data():

    print("Loading PEMS04 data...")

    data = np.load(NPZ_PATH)['data']
    print(f"Traffic data shape: {data.shape}")

    distance_edges = []
    with open(DISTANCE_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            distance_edges.append({
                'from': int(row['from']),
                'to': int(row['to']),
                'cost': float(row['cost'])
            })
    print(f"Distance edges: {len(distance_edges)}")

    return data, distance_edges

def compute_quantile_thresholds(data, feature_idx=0):

    all_values = data[:, :, feature_idx].flatten()

    threshold_low = np.percentile(all_values, 33.33)
    threshold_high = np.percentile(all_values, 66.67)

    print(f"Flow discretization thresholds: low={threshold_low:.2f}, high={threshold_high:.2f}")
    print(f"Flow range: [{all_values.min():.2f}, {all_values.max():.2f}]")

    return [threshold_low, threshold_high]

def discretize_labels(values, thresholds):

    labels = np.zeros_like(values, dtype=np.int64)
    labels[values >= thresholds[0]] = 1
    labels[values >= thresholds[1]] = 2
    return labels

def build_spatial_edges(distance_edges, time_window_offset, num_sensors=NUM_SENSORS):

    src_nodes = []
    dst_nodes = []

    for row in distance_edges:
        src = int(row['from']) + time_window_offset
        dst = int(row['to']) + time_window_offset

        src_nodes.extend([src, dst])
        dst_nodes.extend([dst, src])

    return src_nodes, dst_nodes

def build_temporal_edges(time_window, num_sensors=NUM_SENSORS):

    if time_window == 0:
        return [], []

    src_nodes = []
    dst_nodes = []

    current_offset = time_window * num_sensors
    prev_offset = (time_window - 1) * num_sensors

    for sensor_id in range(num_sensors):
        current_node = sensor_id + current_offset
        prev_node = sensor_id + prev_offset

        src_nodes.extend([current_node, prev_node])
        dst_nodes.extend([prev_node, current_node])

    return src_nodes, dst_nodes

def aggregate_features(data, task_idx, num_tasks=NUM_TASKS):

    T, N, F = data.shape
    steps_per_task = T // num_tasks

    start_idx = task_idx * steps_per_task
    end_idx = start_idx + steps_per_task

    features = data[start_idx:end_idx, :, :].mean(axis=0)

    return features

def build_task_graph(data, distance_edges, task_idx, thresholds, num_tasks=NUM_TASKS):

    T, N, F = data.shape

    total_nodes = (task_idx + 1) * N

    g = dgl.DGLGraph()
    g.add_nodes(total_nodes)

    all_src = []
    all_dst = []

    for t in range(task_idx + 1):
        offset = t * N
        src, dst = build_spatial_edges(distance_edges, offset, N)
        all_src.extend(src)
        all_dst.extend(dst)

    for t in range(1, task_idx + 1):
        src, dst = build_temporal_edges(t, N)
        all_src.extend(src)
        all_dst.extend(dst)

    if all_src:
        g.add_edges(all_src, all_dst)

    all_features = []
    all_labels = []
    node_idxs = []
    new_nodes_mask = []

    for t in range(task_idx + 1):

        features = aggregate_features(data, t, num_tasks)
        all_features.append(features)

        flow_values = features[:, 0]
        labels = discretize_labels(flow_values, thresholds)
        all_labels.append(labels)

        offset = t * N
        node_idxs.extend([offset + i for i in range(N)])

        if t == task_idx:
            new_nodes_mask.extend([True] * N)
        else:
            new_nodes_mask.extend([False] * N)

    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    g.ndata['x'] = torch.tensor(all_features, dtype=torch.float32)
    g.ndata['y'] = torch.tensor(all_labels, dtype=torch.long)
    g.ndata['node_idxs'] = torch.tensor(node_idxs, dtype=torch.long)
    g.ndata['new_nodes_mask'] = torch.tensor(new_nodes_mask, dtype=torch.bool)

    return g

def generate_masks(g, num_runs=10):

    n_nodes = g.num_nodes()
    new_node_idxs = g.ndata['new_nodes_mask'].nonzero().squeeze().numpy()
    n_new_nodes = len(new_node_idxs)

    masks_list = []

    for run in range(num_runs):
        np.random.seed(run)
        shuffled_idx = np.random.permutation(n_new_nodes)

        n_train = int(TRAIN_RATIO * n_new_nodes)
        n_valid = int(VALID_RATIO * n_new_nodes)

        train_idx = new_node_idxs[shuffled_idx[:n_train]]
        valid_idx = new_node_idxs[shuffled_idx[n_train:n_train + n_valid]]
        test_idx = new_node_idxs[shuffled_idx[n_train + n_valid:]]

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        valid_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

        masks_list.append((train_mask, valid_mask, test_mask))

    return masks_list

def main():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data, distance_edges = load_pems04_data()

    thresholds = compute_quantile_thresholds(data, feature_idx=0)

    print(f"\nSaving statistics: num_tasks={NUM_TASKS}, num_classes={NUM_CLASSES}")
    with open(os.path.join(OUTPUT_DIR, 'statistics'), 'wb') as f:
        pickle.dump((NUM_TASKS, NUM_CLASSES), f)

    with open(os.path.join(OUTPUT_DIR, 'thresholds'), 'wb') as f:
        pickle.dump(thresholds, f)

    print("\nBuilding task graphs...")
    all_masks = [[] for _ in range(10)]

    for task_idx in range(NUM_TASKS):
        print(f"\nTask {task_idx}:")

        g = build_task_graph(data, distance_edges, task_idx, thresholds, NUM_TASKS)

        print(f"  Nodes: {g.num_nodes()}, Edges: {g.num_edges()}")
        print(f"  New nodes: {g.ndata['new_nodes_mask'].sum().item()}")
        print(f"  Features shape: {g.ndata['x'].shape}")

        labels = g.ndata['y'].numpy()
        new_mask = g.ndata['new_nodes_mask'].numpy()
        new_labels = labels[new_mask]
        for c in range(NUM_CLASSES):
            count = (new_labels == c).sum()
            print(f"  Class {c} (new nodes): {count} ({100*count/len(new_labels):.1f}%)")

        with open(os.path.join(OUTPUT_DIR, f'sub_graph_{task_idx}_by_edges'), 'wb') as f:
            pickle.dump(g, f)

        masks_list = generate_masks(g, num_runs=10)
        for run, masks in enumerate(masks_list):
            all_masks[run].append(masks)

    print("\nSaving masks...")
    for run in range(10):
        with open(os.path.join(OUTPUT_DIR, f'mask_seed_{run}'), 'wb') as f:
            pickle.dump(all_masks[run], f)

    print("\nDone! Data saved to:", OUTPUT_DIR)
    print(f"\nSummary:")
    print(f"  - {NUM_TASKS} tasks")
    print(f"  - {NUM_SENSORS} sensors per time window")
    print(f"  - {NUM_TASKS * NUM_SENSORS} total spatiotemporal nodes")
    print(f"  - {NUM_CLASSES} classes (discretized flow)")
    print(f"  - Spatial edges: from distance.csv (bidirectional)")
    print(f"  - Temporal edges: bidirectional between adjacent time windows")

if __name__ == '__main__':
    main()

## GMT

Source code for GMT (Graph Mixture-of-experts with Topology-aware coarsening), built on top of the TACO framework for continual graph learning.

## Prerequisites

Python 3.8.13, PyTorch 1.12.1+cu11.6, dgl 0.9.1, pygsp 0.5.1, sklearn 1.1.2

## Getting Started

Place raw data files into the `raw-data` folder.

```
GMT/
  raw-data/
  src/
  DBLP-data_preprocess/
```

### Preprocess

Run the following under the corresponding `[dataset]-data_preprocess/` folder:

```
python 0-process_dataset.py
python 1-build_graph.py
python 2-get_largest_connected_component.py
python 3-generate_random_masks.py
```

### Train

```
python -W ignore train.py --Dataset kindle --method DYGRA --gnn GCN --reduction_rate 0.5 --buffer_size 200
```

With MoE enabled:

```
python -W ignore train.py --Dataset kindle --method DYGRA --gnn GCN --reduction_rate 0.5 --buffer_size 200 --use_moe True --num_experts 4
```

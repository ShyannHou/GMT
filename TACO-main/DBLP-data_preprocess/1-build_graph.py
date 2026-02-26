import json
import csv
from collections import defaultdict
import pickle
import torch
import dgl
import os
import numpy as np

def get_research_area(venue):
    db = {'SIGMOD','ICDE','VLDB','EDBT','PODS','ICDT','DASFAA','SSDBM','CIKM'}
    dm = {'KDD','ICDM','SDM','PKDD','PAKDD'}
    ai = {'IJCAI','AAAI','NIPS','ICML','ECML','ACML','IJCNN','UAI','ECAI','COLT','ACL','KR'}
    cv = {'CVPR','ICCV','ECCV','ACCV','MM','ICPR','ICIP','ICME'}
    if venue.upper() in db: return 0
    if venue.upper() in dm: return 1
    if venue.upper() in ai: return 2
    if venue.upper() in cv: return 3
    return -1

datapath = '../data/DBLP/'
if not os.path.exists(datapath):
    os.makedirs(datapath)
num_task = 10
num_class = 4
with open(datapath+'statistics', 'wb') as file:
  pickle.dump((num_task, num_class), file)

dataset_processed = '../raw-data/dblpv13_processed.json'
keywords_idx_path = os.path.join(datapath, 'keywords.idx')
fos_idx_path = os.path.join(datapath, 'fos.idx')
all_papers_path = os.path.join(datapath, 'all_papers')

def load_cached_artifacts():
    if not (os.path.exists(keywords_idx_path) and os.path.exists(fos_idx_path) and os.path.exists(all_papers_path)):
        return None
    with open(keywords_idx_path, 'rb') as file:
        keywords_idx = pickle.load(file)
    with open(fos_idx_path, 'rb') as file:
        fos_idx = pickle.load(file)
    with open(all_papers_path, 'rb') as file:
        all_papers = pickle.load(file)
    return keywords_idx, fos_idx, all_papers

cached = load_cached_artifacts()
if cached is not None:
    keywords_idx, fos_idx, all_papers = cached
    num_keywords = len(keywords_idx)
    num_fos = len(fos_idx)
    target_papers_all_id = set()
    target_papers_year_id = [set() for _ in range(num_task)]
    for paper_id, paper in all_papers.items():
        if 'class' in paper:
            target_papers_all_id.add(paper_id)
            year = paper['year']
            target_papers_year_id[max((year - 1995) // 2, 0)].add(paper_id)
else:

    target_papers_all_id = set()
    target_papers_year_id = [set() for i in range(num_task)]
    keywords_dic = defaultdict(int)
    fos_dic = defaultdict(int)

    infile = open(dataset_processed)
    n = 0
    paper_all_years = {}
    for x in infile:
        item = json.loads(x)
        if 'keywords' not in item or len(item['keywords']) == 0:
            continue
        if 'fos' not in item or len(item['fos']) == 0:
            continue
        if 'year' not in item or item['year'] > 2014:
            continue
        if 'venue' not in item or 'raw' not in item['venue']:
            continue
        research_area = get_research_area(item['venue']['raw'])
        if research_area < 0:
            continue
        target_papers_all_id.add(item['_id'])
        target_papers_year_id[max((item['year']-1995)//2, 0)].add(item['_id'])

        for kw in item['keywords']:
                keywords_dic[kw] += 1
        for fos in item['fos']:
            fos_dic[fos] += 1

    from copy import copy
    keywords_dic_reduced = copy(keywords_dic)

    for key in keywords_dic:
        if keywords_dic[key] < 500:
            del keywords_dic_reduced[key]

    fos_dic_reduced = copy(fos_dic)

    for key in fos_dic:
        if fos_dic[key] < 500:
            del fos_dic_reduced[key]

    keywords_idx = {}
    fos_idx = {}
    for key in keywords_dic_reduced:
        keywords_idx[key] = len(keywords_idx)
    for key in fos_dic_reduced:
        fos_idx[key] = len(fos_idx)

    num_keywords = len(keywords_dic_reduced)
    num_fos = len(fos_dic_reduced)

    all_papers = {}

    infile = open(dataset_processed)

    paper_all_years = {}
    for x in infile:
        item = json.loads(x)
        if 'keywords' not in item or len(item['keywords']) == 0:
            continue
        if 'fos' not in item or len(item['fos']) == 0:
            continue
        if 'year' not in item or item['year'] > 2014:
            continue
        if 'venue' not in item or 'raw' not in item['venue']:
            continue
        research_area = get_research_area(item['venue']['raw'])
        references = item['references'] if 'references' in item else []

        year = item['year']
        if item['_id'] in target_papers_all_id:
            year = item['year']
            keyword_feat = [0 for i in range(num_keywords)]
            fos_feat = [0 for i in range(num_fos)]
            for kw in item['keywords']:
                if kw in keywords_idx:
                    keyword_feat[keywords_idx[kw]] = 1
                for fos in item['fos']:
                    if fos in fos_idx:
                        fos_feat[fos_idx[fos]] = 1
            paper = {'keyword':keyword_feat, 'fos':fos_feat, 'ref':references, 'class':research_area, 'year':year}

        elif any (ref in target_papers_all_id for ref in references):
            paper = {'ref':[ref for ref in references if (ref in target_papers_all_id)], 'year':year}

        else: continue
        all_papers[item['_id']] = paper

    with open(datapath+'keywords.idx', 'wb') as file:
      pickle.dump(keywords_idx, file)
    with open(datapath+'fos.idx', 'wb') as file:
      pickle.dump(fos_idx, file)

    with open(datapath+'all_papers', 'wb') as file:
      pickle.dump(all_papers, file)

id2idx = {}
for id in target_papers_all_id:
    id2idx[id] = len(id2idx)

if os.getenv("SKIP_FULL_GRAPH") != "1":
    node_features = [[] for i in range(len(id2idx))]

    class_label = [-1 for i in range(len(id2idx))]
    g = dgl.DGLGraph()
    g.add_nodes(len(id2idx))
    n = 0
    for paper_id in id2idx:
        n+=1
        if n%5000 == 0: print (n)
        paper_year = all_papers[paper_id]['year']

        idx = id2idx[paper_id]
        node_features[idx] = all_papers[paper_id]['keyword']+all_papers[paper_id]['fos']
        class_label[idx] = all_papers[paper_id]['class']
        for ref_id in all_papers[paper_id]['ref']:
            if ref_id not in all_papers or ref_id == paper_id or all_papers[ref_id]['year'] > paper_year:
                continue
            if ref_id in id2idx:
                g.add_edges(idx, id2idx[ref_id])
                g.add_edges(id2idx[ref_id], idx)

            else:
                for ref_ref_id in all_papers[ref_id]['ref']:
                    if ref_ref_id in id2idx:
                        g.add_edges(idx, id2idx[ref_ref_id])
                        g.add_edges(id2idx[ref_ref_id], idx)

    node_features = torch.tensor(node_features)
    class_label = torch.tensor(class_label)

    g.ndata['x'] = node_features
    g.ndata['y'] = class_label
    with open(datapath+'graph_full', 'wb') as file:
      pickle.dump(g, file)

g_list = []
for time_slot in range(num_task):
    id2idx_t = {}

    for paper_id in target_papers_year_id[time_slot]:
        id2idx_t[paper_id] = len(id2idx_t)
    num_new_papers = len(id2idx_t)

    for paper_id in target_papers_year_id[time_slot]:
        paper = all_papers[paper_id]
        paper_year = paper['year']
        for ref_id in paper['ref']:
            if ref_id not in all_papers or ref_id == paper_id or all_papers[ref_id]['year'] > paper_year:
                continue
            if ref_id not in id2idx_t and ref_id in id2idx:
                id2idx_t[ref_id] = len(id2idx_t)
            for ref_ref_id in all_papers[ref_id]['ref']:
                if ref_ref_id not in id2idx_t and ref_ref_id in id2idx and all_papers[ref_ref_id]['year']<=paper_year:
                    id2idx_t[ref_ref_id] = len(id2idx_t)

    node_features = [[] for i in range(len(id2idx_t))]
    node_idxs = [-1 for i in range(len(id2idx_t))]
    class_label = [-1 for i in range(len(id2idx_t))]
    g = dgl.DGLGraph()
    g.add_nodes(len(id2idx_t))
    for paper_id in id2idx_t:
        idx = id2idx_t[paper_id]
        node_features[idx] = all_papers[paper_id]['keyword']+all_papers[paper_id]['fos']
        node_idxs[idx] = id2idx[paper_id]
        class_label[idx] = all_papers[paper_id]['class']
    for paper_id in target_papers_year_id[time_slot]:
        paper = all_papers[paper_id]
        idx = id2idx_t[paper_id]
        paper_year = paper['year']
        for ref_id in paper['ref']:
            if ref_id not in all_papers or ref_id == paper_id or all_papers[ref_id]['year'] > paper_year:
                continue
            if ref_id in id2idx_t:
                g.add_edges(idx, id2idx_t[ref_id])
                g.add_edges(id2idx_t[ref_id], idx)
            else:
                for ref_ref_id in all_papers[ref_id]['ref']:
                    if ref_ref_id in id2idx_t:
                        g.add_edges(idx, id2idx_t[ref_ref_id])
                        g.add_edges(id2idx_t[ref_ref_id], idx)
    node_features = torch.tensor(node_features)
    class_label = torch.tensor(class_label)
    node_idxs = torch.tensor(node_idxs)
    g.ndata['num_new_nodes'] = torch.tensor([num_new_papers for i in range(len(id2idx_t))])
    g.ndata['x'] = node_features
    g.ndata['node_idxs'] = node_idxs
    g.ndata['y'] = class_label
    print (g)
    g_list.append(g)
    with open(datapath+f'graph_{time_slot}_by_edges', 'wb') as file:
      pickle.dump(g, file)

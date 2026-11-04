"""AMLSim inputs and a minimal importer for the unmodified MEGA-PNA model.

The official package eagerly imports every dataset/architecture. Namespace-only
packages below avoid unrelated C++ port-numbering and other-model dependencies;
actual model, encoder, head and aggregation source files are imported unchanged.
"""
import os
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import sys
import types
import importlib
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW = Path(r'C:\habib dissertation\Projects\defence\aml_starting_point_framwork\data\raw\ibm_amlsim')
FORWARD = ('node', 'to', 'node')
REVERSE = ('node', 'rev_to', 'node')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(x):
    a = np.ascontiguousarray(x.detach().cpu().numpy())
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def import_model():
    for name in ['MegaGNN', 'MegaGNN.graphgym', 'MegaGNN.graphgym.models',
                 'MegaGNN.graphgym.contrib', 'MegaGNN.graphgym.contrib.layer',
                 'MegaGNN.config', 'MegaGNN.act', 'MegaGNN.encoder', 'MegaGNN.head',
                 'MegaGNN.layer', 'MegaGNN.network', 'MegaGNN.train']:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split('.')))]
            module.__package__ = name
            sys.modules[name] = module
    from MegaGNN.graphgym.config import cfg
    from MegaGNN.graphgym import register
    for name in ['misc_config', 'dataset_config', 'ablation_config']:
        importlib.import_module('MegaGNN.config.' + name)
    for key, func in list(register.config_dict.items()):
        if key != 'cfg_gnn':
            func(cfg)
    importlib.import_module('MegaGNN.act.example')
    importlib.import_module('MegaGNN.encoder.hetero_raw_encoder')
    importlib.import_module('MegaGNN.head.hetero_node')
    from MegaGNN.network.megagnn import GNNEdgeModel
    return cfg, GNNEdgeModel


def configure():
    cfg, cls = import_model()
    cfg.device = 'cpu'
    cfg.dataset.name = 'AMLSim'
    cfg.dataset.task = 'hetero_node'
    cfg.dataset.task_entity = 'node'
    cfg.dataset.node_encoder = True
    cfg.dataset.node_encoder_name = 'Hetero_Raw'
    cfg.dataset.node_encoder_bn = False
    cfg.dataset.edge_encoder = True
    cfg.dataset.edge_encoder_name = 'Hetero_Raw'
    cfg.dataset.edge_encoder_bn = False
    cfg.train.add_ego_id = False
    cfg.train.sampler = 'full_batch'
    cfg.model.type = 'MegaGNNModel'
    from yacs.config import CfgNode
    cfg.share.dim_in = CfgNode({'node': 1, 'node__to__node': 2, 'node__rev_to__node': 2})
    cfg.gnn.layer_type = 'PNA'
    cfg.gnn.dim_inner = 20
    cfg.gnn.layers_mp = 2
    cfg.gnn.layers_pre_mp = 0
    cfg.gnn.head = 'hetero_node_per_neighbor'
    cfg.gnn.layers_post_mp = 2
    cfg.gnn.dropout = 0.1
    cfg.gnn.input_dropout = 0.0
    cfg.gnn.layer_norm = False
    cfg.gnn.batch_norm = True
    cfg.gnn.edge_updates = True
    cfg.gnn.multi_edge_agg = True
    cfg.gnn.multi_edge_agg_type = 'pna'
    cfg.gnn.towers = 5
    cfg.gnn.residual = True
    cfg.ablation.random_grouping = False
    return cfg, cls


def prepare(raw=RAW):
    a = pd.read_csv(raw/'accounts.csv')
    t = pd.read_csv(raw/'transactions.csv')
    ids = a.ACCOUNT_ID.to_numpy(dtype=np.int64)
    assert len(ids) == len(np.unique(ids)) == 10000
    assert a.IS_FRAUD.dtype == bool
    y = a.IS_FRAUD.to_numpy(dtype=np.int64)
    src = pd.Index(ids).get_indexer(t.SENDER_ACCOUNT_ID)
    dst = pd.Index(ids).get_indexer(t.RECEIVER_ACCOUNT_ID)
    assert (src >= 0).all() and (dst >= 0).all()
    assert t.TX_ID.is_unique and len(t) == 1323234
    assert t.TX_TYPE.nunique() == 1  # all TRANSFER; constant type adds no signal
    idx = np.arange(len(a))
    tv, test = train_test_split(idx, test_size=.2, random_state=42, stratify=y)
    train, val = train_test_split(tv, test_size=.15/.8, random_state=42, stratify=y[tv])
    masks = {}
    for key, chosen in [('train', train), ('validation', val), ('test', test), ('all', idx)]:
        mask = torch.zeros(len(a), dtype=torch.bool)
        mask[chosen] = True
        masks[key] = mask
    assert torch.equal(sum(masks[k].long() for k in ['train','validation','test']), torch.ones(len(a),dtype=torch.long))
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    raw_attr = torch.tensor(t[['TIMESTAMP','TX_AMOUNT']].to_numpy(), dtype=torch.float32)
    assert torch.isfinite(raw_attr).all()
    # Same z-normalization convention as upstream; full unlabeled graph is
    # visible in this transductive account split. No labels enter preprocessing.
    mean, std = raw_attr.mean(0), raw_attr.std(0)
    std[std == 0] = 1
    attr = (raw_attr-mean)/std
    group = edge_index[0]*len(a)+edge_index[1]
    graph = HeteroData()
    graph['node'].x = torch.zeros(len(a), 1)  # upstream z_norm(ones)
    graph['node'].y = torch.from_numpy(y)
    graph['node'].num_nodes = len(a)
    graph['node'].batch_size = len(a)
    for rel, edges in [(FORWARD,edge_index),(REVERSE,edge_index.flip(0))]:
        graph[rel].edge_index = edges
        graph[rel].edge_attr = attr.clone()
        graph[rel].simp_edge_batch = group.clone()
    metadata = dict(nodes=len(a), illicit=int(y.sum()), transactions=len(t),
                    distinct_directed_pairs=int(group.unique().numel()),
                    self_transactions=int((src==dst).sum()),
                    edge_features=['TIMESTAMP','TX_AMOUNT'], node_features='constant zero (normalized ones)',
                    feature_mean=mean.tolist(), feature_std=std.tolist(),
                    split_seed=42, split='stratified 65/15/20 account split',
                    raw_files={name:sha(raw/name) for name in ['accounts.csv','transactions.csv']},
                    edge_sha256=tensor_sha(edge_index), edge_attribute_sha256=tensor_sha(attr),
                    label_sha256=tensor_sha(graph['node'].y),
                    splits={k:dict(nodes=int(m.sum()),illicit=int(graph['node'].y[m].sum()),mask_sha256=tensor_sha(m)) for k,m in masks.items()})
    return graph, masks, ids, t.TX_ID.to_numpy(), metadata


def make_model(graph):
    cfg, cls = configure()
    return cls({'node':1}, 2, [graph])


def no_transactions(graph):
    out = graph.clone()
    for rel in out.edge_types:
        out[rel].edge_index = torch.empty(2,0,dtype=torch.long)
        out[rel].edge_attr = torch.empty(0,2)
        out[rel].simp_edge_batch = torch.empty(0,dtype=torch.long)
    return out


def forward(model, graph):
    # Upstream encoders and edge updates mutate their batch in place.
    return model(graph.clone())[0]


if __name__ == '__main__':
    import time
    import psutil
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cfg, cls = configure()
    print('Imports passed', flush=True)
    start=time.perf_counter()
    graph,masks,ids,tx_ids,meta=prepare()
    print(meta, 'preparation_s',time.perf_counter()-start,flush=True)
    torch.manual_seed(42)
    model=make_model(graph)
    print('parameters',sum(p.numel() for p in model.parameters()),flush=True)
    start=time.perf_counter()
    model.eval()
    with torch.no_grad():
        p=forward(model,graph)
        print('full inference',p.shape,'seconds',time.perf_counter()-start,'rss_gb',psutil.Process().memory_info().rss/1e9,flush=True)
        q=forward(model,no_transactions(graph))
        print('empty inference',q.shape,'finite',torch.isfinite(q).all().item(),flush=True)
    model.train()
    start=time.perf_counter()
    logits=forward(model,graph)
    loss=torch.nn.functional.cross_entropy(logits[masks['train']],graph['node'].y[masks['train']])
    loss.backward()
    print('training step seconds',time.perf_counter()-start,'loss',loss.item(),'rss_gb',psutil.Process().memory_info().rss/1e9,flush=True)

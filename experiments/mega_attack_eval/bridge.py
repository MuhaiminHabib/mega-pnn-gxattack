"""GPU execution bridge; reuse the repaired adapter and frozen PyG 2.5.2."""
import sys,os,json,hashlib,time,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPAIR=HERE.parent/'mega_explainer_audit/repair_v2'
sys.path[:0]=[str(HERE/'vendor'),str(REPAIR)]
from transaction_adapter_v2 import *
sys.path.insert(0,str(ROOT/'genagg'))
from torch_geometric.explain import GNNExplainer,ExplainerConfig,ModelConfig
from torch_geometric.explain.algorithm.utils import clear_masks
import torch.nn.functional as F

CHECKPOINT=ROOT/'results/amlsim_edge_audit_v1/best.pt'
SELECTION=REPAIR/'fit/selection.json'
MODEL_HASH='1314ea783d2c05bdc34729f59c3deb1680a374bbe10d8450e3a1c9cfc3d4179a'
DEVICE='cuda'

def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__))));t.replace(p)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def move_context(c,device=DEVICE):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in c.items()}

def subset(c,keep):
    d=dict(c)
    for k in ['edge_index']:d[k]=c[k][:,keep]
    for k in ['edge_attr','tx_rows']:d[k]=c[k][keep]
    return d

def official(c,device='cpu'):
    c=move_context(c,device);g=HeteroData()
    g['node'].x=c['x'];g['node'].y=c['y'];g['node'].num_nodes=len(c['x']);g['node'].batch_size=len(c['x'])
    ei=c['edge_index'];group=ei[0]*len(c['x'])+ei[1]
    for rel,e in [(FORWARD,ei),(REVERSE,ei.flip(0))]:
        g[rel].edge_index=e;g[rel].edge_attr=c['edge_attr'];g[rel].simp_edge_batch=group
    return g

class Study:
    def __init__(self):
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        self.graph,self.masks,self.ids,self.txids,self.meta=prepare()
        self.model=make_model(self.graph)
        self.model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['state_dict']);self.model.eval()
        assert state_sha(self.model)==MODEL_HASH
        self.model.to(DEVICE);self.adapter=TransactionAdapter(self.model)
        selection=json.loads(SELECTION.read_text())
        assert sha(selection['checkpoint'])==selection['checkpoint_sha256']
        for p,h in selection['source_hashes'].items():assert sha(p)==h
        cfg=selection['config'];self.pg=ContextPGExplainer(epochs=cfg['epochs'],lr=.003)
        self.pg.coeffs=dict(edge_size=cfg['edge_size'],edge_ent=1.,temp=cfg['temp'],bias=0.)
        self.pg.connect(ExplainerConfig(explanation_type='phenomenon',edge_mask_type='object',node_mask_type=None),ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
        saved=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
        self.pg.mlp.load_state_dict(saved['mlp']);self.pg._curr_epoch=saved['epoch'];self.pg.to(DEVICE)
        for p in self.pg.parameters():p.requires_grad_(False)
        self.probs=np.load(ROOT/'results/amlsim_edge_audit_v1/predictions.npz')['full_graph']
        self.degree=torch.bincount(self.graph[FORWARD].edge_index.flatten(),minlength=10000).double().numpy()
        self.donors=(self.masks['train'][self.graph[FORWARD].edge_index].all(0)).nonzero().flatten().numpy()
        self.bound=None

    def bind(self,c):
        if self.bound is not c:self.adapter.bind(c);self.bound=c

    def predict(self,c,w=None,embedding=False):
        self.bind(c);self.adapter.direct_weights=w
        captured=[]
        handle=self.adapter.port.register_forward_hook(lambda module,inp,out:captured.append(out)) if embedding else None
        try:
            out=self.adapter(c['x'],c['edge_index'])
            return (out,captured[-1]) if embedding else out
        finally:
            self.adapter.direct_weights=None
            if handle:handle.remove()

    def pg_scores(self,c,w=None):
        logits,z=self.predict(c,w,True)
        scores=self.pg.mlp(self.pg._get_inputs(z,c['edge_index'],c['target'])).flatten().sigmoid()
        return logits,scores

    def explanation(self,c,method,seed):
        seed_all(seed);self.bind(c)
        if method=='pgexplainer':
            with torch.no_grad():_,scores=self.pg_scores(c)
        else:
            algorithm=GNNExplainer(epochs=100,lr=.01)
            algorithm.coeffs=dict(edge_size=.005,edge_reduction='sum',node_feat_size=1.,node_feat_reduction='mean',edge_ent=1.,node_feat_ent=.1,EPS=1e-15)
            algorithm.connect(ExplainerConfig(explanation_type='model',edge_mask_type='object',node_mask_type=None),ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
            labels=torch.ones(len(c['x']),device=DEVICE,dtype=torch.long)
            try: scores=algorithm(self.adapter,c['x'],c['edge_index'],target=labels,index=c['target']).edge_mask.detach()
            finally:clear_masks(self.adapter)
        assert bool(torch.isfinite(scores).all())
        with torch.no_grad():soft=float(self.predict(c,scores)[c['target']].softmax(0)[1])
        ids=c['tx_rows'].detach().cpu().numpy();values=scores.detach().cpu().numpy()
        # Stable ID-derived tie ordering survives edge insertions and deletions.
        tie=np.array([int.from_bytes(hashlib.blake2b(f'{seed}:{int(i)}'.encode(),digest_size=8).digest(),'little') for i in ids],dtype=np.uint64)
        order=np.lexsort((tie,-values))
        top=order[:5];edges=c['nodes'][c['edge_index']].detach().cpu().numpy()
        return dict(top5=ids[top].tolist(),top25=ids[order[:25]].tolist(),top5_pairs=edges[:,top].T.tolist(),
                    top5_scores=values[top].tolist(),soft_probability=soft,soft_label_preserved=soft>.5,
                    mean_mask=float(values.mean()),fifth_score_ties=int((values==values[top[-1]]).sum()),seed=seed)

    def context(self,target,max_additions=256):
        clean=build_context(self.graph,target);c=move_context(clean)
        ei=clean['edge_index'];t=clean['target']
        adjacent=torch.unique(ei[:,(ei[0]==t)|(ei[1]==t)]).tolist()
        present=set(map(tuple,ei.T.tolist()))
        missing=[(u,v) for u in adjacent for v in adjacent if u!=v and (u,v) not in present]
        rng=random.Random(20261016+target);rng.shuffle(missing);missing=missing[:max_additions]
        ae=torch.tensor(missing,dtype=torch.long).reshape(-1,2).T
        donors=np.random.default_rng(20261016+target).choice(self.donors,size=len(missing),replace=True)
        support=dict(clean)
        support['edge_index']=torch.cat([ei,ae],1)
        support['edge_attr']=torch.cat([clean['edge_attr'],self.graph[FORWARD].edge_attr[donors]])
        global_add=clean['nodes'][ae]
        support['tx_rows']=torch.cat([clean['tx_rows'],1323234+global_add[0]*10000+global_add[1]])
        support=move_context(support)
        base=torch.cat([torch.ones(ei.size(1)),torch.zeros(len(missing))]).to(DEVICE)
        return c,support,base,donors.tolist()

    def full_probability(self,support,base,flips):
        """Physical replay using official source, CPU, both transaction relations."""
        m=int(base.sum());delete=[i for i in flips if i<m];add=[i for i in flips if i>=m]
        original=self.graph;g=HeteroData()
        for k in ['x','y','num_nodes','batch_size']:g['node'][k]=original['node'][k]
        keep=torch.ones(original[FORWARD].num_edges,dtype=torch.bool)
        if delete:keep[support['tx_rows'][delete].cpu()]=False
        added=support['nodes'][support['edge_index'][:,add]].cpu()
        attrs=support['edge_attr'][add].cpu()
        ei=torch.cat([original[FORWARD].edge_index[:,keep],added],1)
        ea=torch.cat([original[FORWARD].edge_attr[keep],attrs])
        group=ei[0]*10000+ei[1]
        for rel,e in [(FORWARD,ei),(REVERSE,ei.flip(0))]:
            g[rel].edge_index=e;g[rel].edge_attr=ea;g[rel].simp_edge_batch=group
        self.model.cpu()
        try:
            with torch.no_grad():p=float(forward(self.model,g)[support['global_node']].softmax(0)[1])
        finally:self.model.to(DEVICE)
        return p

def change(before,after):
    a=set(before['top5']);b=set(after['top5']);pa=set(map(tuple,before['top5_pairs']));pb=set(map(tuple,after['top5_pairs']))
    return dict(top5_changed_fraction=1-len(a&b)/len(a),pair_jaccard= len(pa&pb)/len(pa|pb),
                top25_changed_fraction=1-len(set(before['top25'])&set(after['top25']))/len(before['top25']))

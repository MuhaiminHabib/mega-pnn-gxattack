"""Isolated adaptations: train-only PG input scaling and unsaturated ranking."""
import sys,itertools
from pathlib import Path
TASK=Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'mega_explainer_pilot_v3'))
from pilot_core import *
TASK=Path(__file__).resolve().parent
REPAIR_CFG=dict(pg_epochs=60,pg_lr=.003,pg_edge_size=.0003,pg_fit_seeds=[2026101901,2026101902,2026101903],gnn_epochs=1500,normalization='Training-input mean/std, floor1e-6; no clipping',ranking='Raw logits before sigmoid; canonical transaction ID only for exact raw ties')

class RepairStudy(PilotStudy):
    def __init__(self):
        super().__init__();self.normalization=None
    def pg_inputs(self,z,c):
        inputs=super().pg_inputs(z,c)
        if self.normalization is not None:
            inputs=(inputs-self.normalization['mean'].to(inputs.device))/self.normalization['std'].to(inputs.device)
        return inputs
    def new_pg(self,seed):
        pg=super().new_pg(seed);pg.epochs=REPAIR_CFG['pg_epochs']
        return pg
    def describe_raw(self,c,raw,scores,seed,trace=None):
        row=super().describe(c,scores,seed,trace)
        ids=c['tx_rows'].cpu().numpy();values=raw.detach().cpu().numpy();order=np.lexsort((ids,-values));edges=c['nodes'][c['edge_index']].cpu().numpy()
        row.update(top5=ids[order[:5]].tolist(),top25=ids[order[:25]].tolist(),top5_pairs=edges[:,order[:5]].T.tolist(),top5_scores=values[order[:5]].tolist(),fifth_score_ties=int((values==values[order[4]]).sum()),score_semantics='Raw logit for ranking; sigmoid for masking')
        return row
    def explain(self,c,method,repeat=0):
        seed=CFG['explanation_seed']+c['global_node']+CFG['repeat_offsets'][repeat]
        self.bind(c);seed_all(seed)
        if method=='pgexplainer':
            with torch.no_grad():
                _,z=self.predict(c,embedding=True);raw=self.pg_fits[repeat].mlp(self.pg_inputs(z,c)).flatten();scores=raw.sigmoid()
            return scores,self.describe_raw(c,raw,scores,seed)
        from gnn_convergence import ConvergedGNN
        a=ConvergedGNN(c['tx_rows'].cpu().numpy(),seed)
        a.connect(ExplainerConfig(explanation_type='model',edge_mask_type='object',node_mask_type=None),ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
        labels=torch.ones(len(c['x']),device=DEVICE,dtype=torch.long)
        try:scores=a(self.adapter,c['x'],c['edge_index'],target=labels,index=c['target']).edge_mask.detach()
        finally:clear_masks(self.adapter)
        raw=torch.from_numpy(a.saved_raw).to(DEVICE);raw[~torch.from_numpy(a.saved_hard).to(DEVICE)]=-torch.inf
        row=self.describe_raw(c,raw,scores,seed,a.trace);row['milestones']=a.milestones
        return scores,row

def load_pg(s):
    folder=TASK/'pg_normalized'
    s.normalization=torch.load(folder/'normalization.pt',map_location='cpu',weights_only=False)
    for seed in REPAIR_CFG['pg_fit_seeds']:
        saved=torch.load(folder/f'pg_{seed}.pt',map_location=DEVICE,weights_only=False)
        assert saved['epoch']==REPAIR_CFG['pg_epochs']-1
        pg=s.new_pg(seed);pg.mlp.load_state_dict(saved['mlp']);pg.mlp.eval();pg._curr_epoch=saved['epoch']
        for p in pg.parameters():p.requires_grad_(False)
        s.pg_fits.append(pg)
    s.pg=s.pg_fits[0]

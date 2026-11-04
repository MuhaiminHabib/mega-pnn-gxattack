"""One convergence experiment: native GNNExplainer objective, frozen detector."""
import sys,json,itertools,time,traceback,msvcrt,gc
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'mega_explainer_pilot_v3'))
from pilot_core import *
BASE=Path(__file__).resolve().parent
OUT=BASE/'gnn_convergence'
STEPS=1500

class ConvergedGNN(IdentityGNN):
    def __init__(self,ids,seed):
        super().__init__(ids,seed);self.epochs=STEPS;self.milestones={}
    def _loss(self,y_hat,y):
        if len(self.trace) in [300,600,1000]:self.capture(len(self.trace))
        return super()._loss(y_hat,y)
    def capture(self,epoch):
        raw=self.edge_mask.detach().cpu().numpy().copy()
        hard=self.hard_edge_mask.detach().cpu().numpy()
        raw[~hard]=-np.inf
        ix=np.lexsort((self.ids,-raw))[:5]
        self.milestones[epoch]=dict(top5=self.ids[ix].tolist(),loss_last=float(self.trace[-1]),mean_mask=float(self.edge_mask.sigmoid().detach().mean()))
    def _clean_model(self,model):
        self.capture(self.epochs)
        self.saved_raw=self.edge_mask.detach().cpu().numpy().copy()
        self.saved_hard=self.hard_edge_mask.detach().cpu().numpy().copy()
        return super()._clean_model(model)

def main():
    OUT.mkdir(exist_ok=True)
    lock=open(OLD/'worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((OLD/'results_v1/status.json').read_text())['state']=='paused'
    s=PilotStudy();old=Path(sys.modules['pilot_core'].__file__).parent/'results'
    nodes=json.loads((old/'protocol.json').read_text())['cohorts']['validation_pilot']
    dump(OUT/'protocol.json',dict(nodes=nodes,epochs=STEPS,lr=.01,objective='Unchanged native GNNExplainer',ranking='raw logits, native hard mask',detector_sha=sha(CHECKPOINT),source_sha=sha(__file__),scope='Development convergence diagnostic on previously exposed validation accounts. No test queries.'))
    for node in nodes:
        c,support,base,_=s.context(node);records=[];m=len(c['tx_rows'])
        for repeat in range(3):
            path=OUT/f'{node}_{repeat}.json'
            if path.exists():records.append(json.loads(path.read_text()));continue
            seed=CFG['explanation_seed']+node+CFG['repeat_offsets'][repeat];seed_all(seed);s.bind(c)
            a=ConvergedGNN(c['tx_rows'].cpu().numpy(),seed)
            a.connect(ExplainerConfig(explanation_type='model',edge_mask_type='object',node_mask_type=None),ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
            labels=torch.ones(len(c['x']),device=DEVICE,dtype=torch.long)
            t=time.time()
            try: scores=a(s.adapter,c['x'],c['edge_index'],target=labels,index=c['target']).edge_mask.detach()
            finally: clear_masks(s.adapter)
            with torch.no_grad():soft=float(s.predict(c,scores)[c['target']].softmax(0)[1])
            ids=c['tx_rows'].cpu().numpy();raw=a.saved_raw.copy();raw[~a.saved_hard]=-np.inf;ix=np.lexsort((ids,-raw))[:5]
            keep=torch.ones(m,dtype=torch.bool,device=DEVICE);keep[ix]=False
            with torch.no_grad():after=float(forward(s.model,official(subset(c,keep),DEVICE))[c['target']].softmax(0)[1])
            record=dict(node=node,repeat=repeat,seed=seed,top5=ids[ix].tolist(),p_clean=float(s.probs[node]),soft=soft,drop_pp=100*(s.probs[node]-after),p_after=after,mean_mask=float(scores.mean()),milestones=a.milestones,loss_tail_change=a.trace[-100]-a.trace[-1],seconds=time.time()-t)
            np.savez_compressed(path.with_suffix('.npz'),raw=a.saved_raw,hard=a.saved_hard,scores=scores.cpu().numpy(),ids=ids)
            dump(path,record);records.append(record);print(path.read_text(),flush=True)
            del a,scores;gc.collect();torch.cuda.empty_cache()
        print('ACCOUNT',node,'replacement',np.mean([1-len(set(a['top5'])&set(b['top5']))/5 for a,b in itertools.combinations(records,2)]),flush=True)
    rows=[json.loads((OUT/f'{n}_{r}.json').read_text()) for n in nodes for r in range(3)]
    result=dict(masked_preserved=sum(r['soft']>.5 for r in rows),n=24,mean_mask=float(np.mean([r['mean_mask'] for r in rows])),primary_drop_pp=float(np.mean([r['drop_pp'] for r in rows if r['repeat']==0])),replacement=float(np.mean([1-len(set(a['top5'])&set(b['top5']))/5 for n in nodes for a,b in itertools.combinations([r for r in rows if r['node']==n],2)])),model_unchanged=state_sha(s.model)==MODEL_HASH)
    dump(OUT/'summary.json',result);print('SUMMARY',json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:dump(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise

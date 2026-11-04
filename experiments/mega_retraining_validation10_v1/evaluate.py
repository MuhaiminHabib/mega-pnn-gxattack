"""Accepted native4500-step explainer, on fixed clean and edited validation graphs."""
import sys,traceback,time,itertools
from common import *
from traced import algorithm_class
def main(arm,node):
    p,lock,guard,pc,attacks,s=start(f'GNN evaluation {arm} account{node}')
    torch,np=pc.torch,pc.np
    assert node in p['cohort']['validation_nodes'] and bool(s.masks['validation'][node]) and not bool(s.masks['test'][node])
    s.model.load_state_dict(state_for(pc,arm));s.model.eval();model_hash=pc.state_sha(s.model)
    assert all(not par.requires_grad for par in s.model.parameters())
    bank=read(ROOT/f'bank/{node}.json');saved=torch.load(ROOT/f'bank/{node}_contexts.pt',map_location='cpu',weights_only=False)
    c=pc.move_context(saved['clean']);support=pc.move_context(saved['support']);base=saved['base'].to(pc.DEVICE)
    with torch.no_grad():p_clean=float(pc.forward(s.model,pc.official(c,pc.DEVICE))[c['target']].softmax(0)[1])
    graph_rows={}
    for variant in ['clean','attack','random','ordinary']:
        if variant!='clean' and not bank[variant]['valid']:
            graph_rows[variant]=dict(valid_bank=False);continue
        flips=[] if variant=='clean' else bank[variant]['flips']
        d=pc.subset(support,attacks.weights(base,flips).bool())
        with torch.no_grad():
            physical=float(pc.forward(s.model,pc.official(d,pc.DEVICE))[d['target']].softmax(0)[1])
            adapted=float(s.predict(d)[d['target']].softmax(0)[1])
        full=None
        if variant!='ordinary':
            full=s.full_probability(support,base,flips)
            assert abs(full-physical)<5e-5
        assert abs(adapted-physical)<5e-5
        graph_rows[variant]=dict(valid_bank=True,p=physical,illicit=physical>.5,confidence_drift_pp=100*(physical-p_clean),
            binary_replay_error=abs(adapted-physical),full_graph_probability=full,flips=flips)
    incident=(c['edge_index'][0]==c['target'])|(c['edge_index'][1]==c['target'])
    with torch.no_grad():isolated=float(pc.forward(s.model,pc.official(pc.subset(c,~incident),pc.DEVICE))[c['target']].softmax(0)[1])
    write(ROOT/f'explanations/{arm}_{node}_graphs.json',dict(arm=arm,node=node,graphs=graph_rows,isolated_probability=isolated,
        original_illicit=p_clean>.5,state_sha256=model_hash,protocol_sha256=sha(ROOT/'protocol.json')))
    for variant in ['clean','attack','random']:
        repeats=range(3) if variant=='clean' else range(1)
        for repeat in repeats:
            name=f'{arm}_{node}_{variant}_{repeat}';dest=ROOT/f'explanations/{name}.json'
            if dest.exists():continue
            common=dict(arm=arm,node=node,variant=variant,repeat=repeat,state_sha256=model_hash,protocol_sha256=sha(ROOT/'protocol.json'))
            if not graph_rows[variant]['valid_bank'] or p_clean<=.5 or not graph_rows[variant]['illicit']:
                write(dest,dict(**common,eligible=False,reason='Missing valid bank edit or clean/edited illicit prediction lost'))
                continue
            flips=graph_rows[variant]['flips'];d=pc.subset(support,attacks.weights(base,flips).bool())
            seed=20261017+node+repeat*100000;pc.seed_all(seed);s.bind(d)
            ids=d['tx_rows'].cpu().numpy();started=time.time()
            def callback(a,step,raw,hard,snap):status(stage='gnn_evaluation',arm=arm,node=node,variant=variant,repeat=repeat,step=step,total_steps=4500)
            Alg=algorithm_class(pc);a=Alg(ids,seed,4500,ROOT/f'checkpoints/{name}.pt',callback)
            a.connect(pc.ExplainerConfig(explanation_type='model',edge_mask_type='object',node_mask_type=None),
                pc.ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
            labels=torch.ones(len(d['x']),device=pc.DEVICE,dtype=torch.long)
            try:scores=a(s.adapter,d['x'],d['edge_index'],target=labels,index=d['target']).edge_mask.detach()
            finally:pc.clear_masks(s.adapter)
            raw=a.saved_raw.copy();hard=a.saved_hard.copy();rank=raw.copy();rank[~hard]=-np.inf
            order=np.lexsort((ids,-rank));keep=torch.ones(len(ids),device=pc.DEVICE,dtype=torch.bool);keep[order[:5]]=False
            with torch.no_grad():
                p_soft=float(s.predict(d,scores)[d['target']].softmax(0)[1])
                p_delete=float(pc.forward(s.model,pc.official(pc.subset(d,keep),pc.DEVICE))[d['target']].softmax(0)[1])
                p_only=float(pc.forward(s.model,pc.official(pc.subset(d,~keep),pc.DEVICE))[d['target']].softmax(0)[1])
            raw_path=ROOT/f'explanations/{name}.npz';tmp=raw_path.with_suffix('.tmp')
            with tmp.open('wb') as stream:np.savez_compressed(stream,ids=ids,raw=raw,hard=hard,scores=scores.cpu().numpy())
            tmp.replace(raw_path)
            write(dest,dict(**common,eligible=True,reused=False,seed=seed,top5=ids[order[:5]].tolist(),top25=ids[order[:25]].tolist(),
                raw_file=str(raw_path),raw_sha256=sha(raw_path),p_graph=graph_rows[variant]['p'],p_soft=p_soft,p_delete=p_delete,
                p_keep_only_top5=p_only,drop_pp=100*(graph_rows[variant]['p']-p_delete),steps=4500,seconds=time.time()-started))
            assert pc.state_sha(s.model)==model_hash
            print('EXPLANATION COMPLETE',name,flush=True)
            del a,scores,d;import gc;gc.collect();torch.cuda.empty_cache()
    assert pc.state_sha(s.model)==model_hash
    write(ROOT/f'completed/eval_{arm}_{node}.json',dict(model_unchanged=True,arm=arm,node=node,model_hash=model_hash))
    validate();guard.close()
if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]))
    except BaseException:write(ROOT/'failure.json',dict(stage='evaluate',traceback=traceback.format_exc()));raise

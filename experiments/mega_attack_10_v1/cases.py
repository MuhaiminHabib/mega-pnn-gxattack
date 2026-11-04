"""Audited confirmation; three separately reported attack/control arms.

Copied from the previous fixed pilot. Changes: dynamic account count, atomic
raw-score cache, persistent status, and PG results evaluated before slow GNN
proposal selection. Attack algorithms/objectives/selection remain unchanged.
"""
import gc
import sys
from pathlib import Path
TASK = Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'mega_priority_repair'))
from repair_core import *
TASK = Path(__file__).resolve().parent
sys.path.insert(0,str(TASK))
import audited_attacks
OUT = TASK/'results'


def progress(**kw):
    try:
        dump(OUT/'status.json',dict(state='running',pid=os.getpid(),updated=time.time(),completed=len(list((OUT/'records').glob('*.json'))),expected=40,**kw))
    except PermissionError:
        pass


def save_scores(path,**arrays):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    tmp.replace(path)


def run_cases(s,protocol,only_node=None):
    ph=sha(OUT/'protocol.json')
    for ordinal,node in enumerate(protocol['cohorts']['validation_pilot']):
        if only_node is not None and node != only_node: continue
        progress(stage='context',account=ordinal+1,total_accounts=len(protocol['cohorts']['validation_pilot']),node=node)
        c,support,base,donors=s.context(node);m=int(base.sum());cache={}
        domain=dict(node=node,original_transactions=m,added_ids=support['tx_rows'][m:].cpu().tolist(),
                    added_pairs=support['nodes'][support['edge_index'][:,m:]].T.cpu().tolist(),donor_csv_rows=donors,protocol_sha256=ph)
        dump(OUT/'domains'/f'{node}.json',domain)
        def explain(method,flips,repeat=0):
            key=f'{node}_{method}_{repeat}_'+hashlib.sha256(json.dumps(sorted(flips)).encode()).hexdigest()[:20]
            p=OUT/'explanations'/(key+'.json');scores_path=p.with_suffix('.npz')
            if key in cache:return cache[key]
            if p.exists():
                result=json.loads(p.read_text());assert result['protocol_sha256']==ph
            else:
                progress(stage='explanation',node=node,explainer=method,repeat=repeat,edits=len(flips))
                d=subset(support,original_attacks.weights(base,flips).bool())
                scores,result=s.explain(d,method,repeat)
                result.update(protocol_sha256=ph,flips=sorted(flips))
                save_scores(scores_path,ids=d['tx_rows'].cpu().numpy(),scores=scores.cpu().numpy(),raw=s.last_raw)
                dump(p,result)
                print('EXPLANATION_SAVED',node,method,repeat,len(flips),flush=True)
            cache[key]=result;return result
        clean={e:[explain(e,[],r) for r in range(3)] for e in ['pgexplainer','gnnexplainer']}
        quality_path=OUT/'quality'/f'{node}.json'
        if not quality_path.exists():
            quality=dict(node=node,split='validation',p_clean=float(s.probs[node]),clean=clean,protocol_sha256=ph)
            quality['repeat_changes']={e:[change(clean[e][a],clean[e][b]) for a,b in itertools.combinations(range(3),2)] for e in clean}
            quality['same_seed_checks']={}
            if ordinal==0:
                for method in clean:
                    d=subset(support,base.bool());scores,again=s.explain(d,method,0)
                    key=f'{node}_{method}_0_'+hashlib.sha256(json.dumps([]).encode()).hexdigest()[:20]
                    saved=np.load(OUT/'explanations'/(key+'.npz'))['scores']
                    error=float(np.max(np.abs(saved-scores.cpu().numpy())))
                    perm=np.random.default_rng(20261017).permutation(m);reordered=dict(d)
                    reordered['edge_index']=d['edge_index'][:,perm]
                    for k in ['edge_attr','tx_rows']:reordered[k]=d[k][perm]
                    ps,pr=s.explain(reordered,method,0)
                    recovered=np.empty(m);recovered[perm]=ps.cpu().numpy()
                    quality['same_seed_checks'][method]=dict(repeat_score_error=error,repeat_change=change(clean[method][0],again),
                        permutation_score_error=float(np.max(np.abs(saved-recovered))),permutation_change=change(clean[method][0],pr))
            rng=random.Random(CFG['selection_seed']+node)
            random_sets=[rng.sample(range(m),5) for _ in range(CFG['removal_random_draws'])]
            random_probs=[]
            for j,ix in enumerate(random_sets):
                keep=torch.ones(m,device=DEVICE,dtype=torch.bool);keep[ix]=False
                with torch.no_grad():p=float(forward(s.model,official(subset(c,keep),DEVICE))[c['target']].softmax(0)[1])
                random_probs.append(p)
            random_full=s.full_probability(c,torch.ones(m,device=DEVICE),random_sets[0]);assert abs(random_full-random_probs[0])<5e-5
            quality['random_removal']=dict(indices=random_sets,probabilities=random_probs,full_replay_first=random_full)
            quality['selected_removal']={}
            mapping={int(tx):i for i,tx in enumerate(c['tx_rows'].cpu().tolist())}
            for method in clean:
                ix=[mapping[tx] for tx in clean[method][0]['top5']]
                keep=torch.ones(m,device=DEVICE,dtype=torch.bool);keep[ix]=False
                with torch.no_grad():local=float(forward(s.model,official(subset(c,keep),DEVICE))[c['target']].softmax(0)[1])
                full=s.full_probability(c,torch.ones(m,device=DEVICE),ix);assert abs(full-local)<5e-5
                quality['selected_removal'][method]=dict(transaction_ids=clean[method][0]['top5'],p_after=full,drop_pp=100*(s.probs[node]-full),full_replay_error=abs(full-local))
            dump(quality_path,quality);print('QUALITY',ordinal+1,node,quality['repeat_changes'],flush=True)
        def query(method,flips,seed):
            assert seed==CFG['explanation_seed']+node
            return explain(method,flips,0)
        def update(**kw):progress(node=node,**kw)
        def pack(best,method):
            if best is None or not best['valid']:
                return dict(valid=False,reason='No valid proposal' if best is None else 'Prespecified random draw failed constraints',
                            attempted=None if best is None else best,change=dict(top5_changed_fraction=0.,pair_jaccard=1.))
            flips=best['flips'];after=[explain(method,flips,r) for r in range(3)]
            progress(stage='full_graph_replay',node=node,explainer=method,edits=len(flips))
            full=s.full_probability(support,base,flips);assert abs(full-best['p'])<5e-5 and full>.5
            edits=[]
            for i in flips:
                u,v=support['nodes'][support['edge_index'][:,i]].cpu().tolist()
                edits.append(dict(operation='delete' if i<m else 'add',transaction_key=int(support['tx_rows'][i]),source=u,destination=v,donor_csv_row=donors[i-m] if i>=m else None))
            added={x['transaction_key'] for x in edits if x['operation']=='add'}
            return dict(valid=True,p_before=float(s.probs[node]),p_after=full,confidence_change_pp=100*(full-float(s.probs[node])),
                        full_graph_replay_error=abs(full-best['p']),flips=flips,edits=edits,actual_edits=len(flips),after=after,
                        change=change(clean[method][0],after[0]),repeat_changes=[change(a,b) for a,b in zip(clean[method],after)],
                        added_top5=sorted(added&set(after[0]['top5'])))
        for family,method in [('deduction','pgexplainer'),('gx','pgexplainer'),('gx','gnnexplainer'),('deduction','gnnexplainer')]:
            record_path=OUT/'records'/f'{family}_{method}_{node}.json'
            if record_path.exists():assert json.loads(record_path.read_text())['protocol_sha256']==ph;continue
            progress(stage='attack',node=node,family=family,explainer=method)
            audit_path=OUT/'audits'/f'{family}_{method if family=="deduction" else "shared"}_{node}.json'
            if audit_path.exists():audit=json.loads(audit_path.read_text());assert audit['protocol_sha256']==ph
            else:
                audit=audited_attacks.deduction(s,c,support,base,clean[method][0],method,query,update) if family=='deduction' else audited_attacks.gx(s,support,base,update)
                audit['protocol_sha256']=ph;dump(audit_path,audit)
            row=dict(node=node,split='validation',family=family,explainer=method,
                     mode='PG-TX-targeted transfer' if family=='gx' and method=='gnnexplainer' else 'direct',
                     clean=clean[method],attack=pack(audit['attack'],method),random=pack(audit['random'],method),ordinary=pack(audit['ordinary'],method),protocol_sha256=ph)
            dump(record_path,row);print('ATTACK',node,family,method,row['attack']['change'],row['random']['change'],flush=True)
            from analyse import analyse
            analyse()
        assert state_sha(s.model)==MODEL_HASH
        del c,support,base,cache;gc.collect();torch.cuda.empty_cache();s.bound=None

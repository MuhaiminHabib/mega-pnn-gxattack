"""Stage1: numerical ranking and adapter audit. No fitting or detector changes."""
import sys,itertools,json,msvcrt,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'mega_explainer_pilot_v3'))
from pilot_core import *
BASE=Path(__file__).resolve().parent
OUT=BASE/'diagnostic'

def main():
    OUT.mkdir(exist_ok=True)
    lock=open(OLD/'worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((OLD/'results_v1/status.json').read_text())['state']=='paused'
    s=PilotStudy();old=Path(pcfile:=sys.modules['pilot_core'].__file__).parent/'results'
    for fit in CFG['pg_fit_seeds']:
        pg=s.new_pg(fit);pg.mlp.load_state_dict(torch.load(old/f'pg_tx_{fit}.pt',map_location=DEVICE,weights_only=False)['mlp']);pg.mlp.eval()
        s.pg_fits.append(pg)
    nodes=json.loads((old/'protocol.json').read_text())['cohorts']['validation_pilot']
    manifest={str(p):sha(p) for p in [Path(__file__),CHECKPOINT,*old.glob('pg_tx_*.pt')]}
    dump(OUT/'protocol.json',dict(nodes=nodes,hashes=manifest,scope='Frozen-fit diagnostic; rank by raw logits instead of rounded sigmoid. No optimization.'))
    records=[]
    for node in nodes:
        c,support,base,_=s.context(node);m=len(c['tx_rows']);ids=c['tx_rows'].cpu().numpy();row=dict(node=node,transactions=m,fits=[],adapter=[])
        with torch.no_grad():
            clean,z=s.predict(c,embedding=True);p=float(clean[c['target']].softmax(0)[1]);inputs=s.pg_inputs(z,c)
            for fi,pg in enumerate(s.pg_fits):
                raw=pg.mlp(inputs).flatten();scores=raw.sigmoid();ix=np.lexsort((ids,-raw.cpu().numpy()))[:5];oldix=np.lexsort((ids,-scores.cpu().numpy()))[:5]
                keep=torch.ones(m,dtype=torch.bool,device=DEVICE);keep[ix]=False
                after=float(forward(s.model,official(subset(c,keep),DEVICE))[c['target']].softmax(0)[1])
                soft=float(s.predict(c,scores)[c['target']].softmax(0)[1])
                row['fits'].append(dict(raw_top5=ids[ix].tolist(),old_top5=ids[oldix].tolist(),saturated=int((scores==1).sum()),raw_top5_scores=raw[ix].cpu().tolist(),soft=soft,delete_probability=after,delete_drop_pp=100*(p-after)))
            for label,w in [('ones',torch.ones_like(base)),('zeros',torch.zeros_like(base)),('base',base),('sparse', (torch.rand_like(base)<.05).float()),('half',(torch.rand_like(base)<.5).float())]:
                adapter=s.predict(support,w)[c['target']].softmax(0)[1]
                physical=forward(s.model,official(subset(support,w.bool()),DEVICE))[c['target']].softmax(0)[1]
                error=float(abs(adapter-physical));assert error<5e-5,(node,label,error)
                row['adapter'].append(dict(mask=label,error=error))
            first=s.pg_fits[0].mlp(inputs).flatten()
            assert torch.equal(first,s.pg_fits[0].mlp(inputs).flatten())
        row['p_clean']=p
        row['raw_replacement']=float(np.mean([1-len(set(a['raw_top5'])&set(b['raw_top5']))/5 for a,b in itertools.combinations(row['fits'],2)]))
        row['old_replacement']=float(np.mean([1-len(set(a['old_top5'])&set(b['old_top5']))/5 for a,b in itertools.combinations(row['fits'],2)]))
        records.append(row);dump(OUT/f'{node}.json',row);print(json.dumps(row),flush=True)
    assert state_sha(s.model)==MODEL_HASH and all(sha(p)==h for p,h in manifest.items())
    result=dict(passed=True,old_replacement=float(np.mean([r['old_replacement'] for r in records])),raw_replacement=float(np.mean([r['raw_replacement'] for r in records])),masked_label_preserved=sum(f['soft']>.5 for r in records for f in r['fits']),raw_deletion_drop_pp=float(np.mean([r['fits'][0]['delete_drop_pp'] for r in records])),max_adapter_error=max(t['error'] for r in records for t in r['adapter']),rows=records)
    dump(OUT/'summary.json',result);print('SUMMARY',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:dump(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise

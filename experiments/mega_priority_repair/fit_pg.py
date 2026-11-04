"""One fixed normalized PG-TX fit, three independent seeds. Detector frozen."""
import gc,msvcrt,traceback
from repair_core import *
OUT=TASK/'pg_normalized'

def main():
    OUT.mkdir(exist_ok=True)
    lock=open(OLD/'worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((OLD/'results_v1/status.json').read_text())['state']=='paused'
    s=RepairStudy();old=TASK.parent/'mega_explainer_pilot_v3/results'
    original=json.loads((old/'protocol.json').read_text())
    for p,h in original['source_hashes'].items():assert sha(p)==h,p
    protocol=dict(config=REPAIR_CFG,training=original['cohorts']['training'],development=original['cohorts']['validation_pilot'],hashes={str(p):sha(p) for p in [Path(__file__),TASK/'repair_core.py',TASK/'gnn_convergence.py',CHECKPOINT]},adaptations='Same PG-TX inputs and native CE/size/entropy loss. Standardize inputs from training accounts only;60 fixed epochs instead of30. No test queries, validation selection, detector training, consistency objective or attack tuning.')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:dump(OUT/'protocol.json',protocol)
    contexts={};inputs={};labels={};total=None;total2=None;count=0
    for node in protocol['training']:
        c=build_context(s.graph,node);contexts[node]=c;d=move_context(c)
        with torch.no_grad():
            logits,z=s.predict(d,embedding=True);v=s.pg_inputs(z,d).detach().cpu();labels[node]=int(logits[d['target']].argmax())
        inputs[node]=v;v=v.double();part=v.sum(0);part2=v.square().sum(0)
        total=part if total is None else total+part;total2=part2 if total2 is None else total2+part2;count+=len(v)
    mu=total/count;std=(total2/count-mu.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    normalization=dict(mean=mu.float(),std=std.float(),training_rows=count)
    if (OUT/'normalization.pt').exists():
        saved=torch.load(OUT/'normalization.pt',weights_only=False);assert torch.equal(saved['mean'],normalization['mean']) and torch.equal(saved['std'],normalization['std'])
    else:torch.save(normalization,OUT/'normalization.pt')
    s.normalization=normalization
    inputs={n:(v-normalization['mean'])/normalization['std'] for n,v in inputs.items()}
    dump(OUT/'input_scaling.json',dict(rows=count,mean=mu.tolist(),std=std.tolist(),training_accounts=len(inputs)))
    ph=sha(OUT/'protocol.json')
    for seed in REPAIR_CFG['pg_fit_seeds']:
        pg=s.new_pg(seed);saved_path=OUT/f'pg_{seed}.pt';history=[];start=0
        if saved_path.exists():
            saved=torch.load(saved_path,map_location=DEVICE,weights_only=False);assert saved['protocol']==ph
            pg.mlp.load_state_dict(saved['mlp']);pg.optimizer.load_state_dict(saved['optimizer']);history=saved['history'];start=saved['epoch']+1
            torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        for epoch in range(start,REPAIR_CFG['pg_epochs']):
            losses=[];ces=[];masks=[];start_time=time.time()
            for node in np.random.default_rng(seed+epoch).permutation(protocol['training']):
                node=int(node);c=move_context(contexts[node]);s.bind(c);pg.optimizer.zero_grad()
                raw=pg.mlp(inputs[node].to(DEVICE)).flatten();sample=pg._concrete_sample(raw,pg._get_temperature(epoch));w=sample.sigmoid()
                output=s.predict(c,w)[c['target']][None];target=torch.tensor([labels[node]],device=DEVICE)
                loss=pg._loss(output,target,sample);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in pg.parameters());pg.optimizer.step()
                losses.append(float(loss.detach()));ces.append(float(F.cross_entropy(output,target).detach()));masks.append(float(w.mean().detach()))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),prediction_loss=float(np.mean(ces)),mask=float(np.mean(masks)),seconds=time.time()-start_time))
            tmp=saved_path.with_suffix('.tmp');torch.save(dict(epoch=epoch,mlp=pg.mlp.state_dict(),optimizer=pg.optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,protocol=ph),tmp);tmp.replace(saved_path)
            dump(OUT/'status.json',dict(state='running',fit=seed,epoch=epoch+1,total_epochs=REPAIR_CFG['pg_epochs']))
            print('FIT',seed,history[-1],flush=True)
        pg.mlp.eval()
        for p in pg.parameters():p.requires_grad_(False)
        s.pg_fits.append(pg)
    rows=[]
    for node in protocol['development']:
        c,*_=s.context(node);r=dict(node=node,explanations=[])
        for repeat in range(3):
            _,e=s.explain(c,'pgexplainer',repeat);r['explanations'].append(e)
        r['replacement']=float(np.mean([change(a,b)['top5_changed_fraction'] for a,b in itertools.combinations(r['explanations'],2)]));rows.append(r);dump(OUT/f'development_{node}.json',r)
    result=dict(n=24,masked_preserved=sum(e['soft_label_preserved'] for r in rows for e in r['explanations']),mean_mask=float(np.mean([e['mean_mask'] for r in rows for e in r['explanations']])),replacement=float(np.mean([r['replacement'] for r in rows])),frozen_detector=state_sha(s.model)==MODEL_HASH)
    assert result['frozen_detector'] and all(sha(p)==h for p,h in protocol['hashes'].items())
    dump(OUT/'summary.json',result);dump(OUT/'status.json',dict(state='complete'));print('SUMMARY',json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:dump(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise

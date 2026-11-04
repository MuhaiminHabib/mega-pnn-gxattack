"""Check the GPU bridge against physical full-graph edits before any attacks."""
from bridge import *
def main():
    s=Study();census=json.loads((HERE/'census.json').read_text())
    targets=sorted((r for r in census['rows'] if r['correctly_detected_illicit'] and r['split']=='train'),key=lambda r:r['context_transactions'])
    cases=[targets[0],targets[len(targets)//2],targets[-1]];records=[]
    for meta in cases:
        node=meta['node'];start=time.perf_counter();c,support,base,donors=s.context(node)
        with torch.no_grad():p=float(s.predict(c)[c['target']].softmax(0)[1]);pbase=float(s.predict(support,base)[c['target']].softmax(0)[1])
        assert abs(p-s.probs[node])<5e-5 and abs(p-pbase)<5e-5
        m=int(base.sum());trials=[[0],[m] if len(base)>m else [1],list(range(5))]
        checks=[]
        for edits in trials:
            w=base.clone();w[edits]=1-w[edits]
            with torch.no_grad():local=float(s.predict(support,w)[c['target']].softmax(0)[1])
            full=s.full_probability(support,base,edits)
            assert abs(full-local)<5e-5,(node,edits,full,local)
            checks.append(dict(edits=edits,local=local,full=full,error=abs(full-local)))
        w=(base*.9+.05).requires_grad_(True)
        logits,scores=s.pg_scores(support,w)
        value=logits[c['target'],1]+scores.mean()
        grad=torch.autograd.grad(value,w)[0];assert torch.isfinite(grad).all() and grad.abs().max()>0
        idx=int(grad.abs().argmax());eps=.001
        wp=w.detach().clone();wm=wp.clone();wp[idx]+=eps;wm[idx]-=eps
        with torch.no_grad():
            lp,sp=s.pg_scores(support,wp);lm,sm=s.pg_scores(support,wm)
            finite=float(((lp[c['target'],1]+sp.mean())-(lm[c['target'],1]+sm.mean()))/(2*eps))
        automatic=float(grad[idx]);assert abs(finite-automatic)<.02+.15*abs(finite),(finite,automatic)
        records.append(dict(node=node,transactions=meta['context_transactions'],clean_error=abs(p-s.probs[node]),
            base_error=abs(p-pbase),physical_checks=checks,gradient_index=idx,autograd=automatic,finite_difference=finite,
            seconds=time.perf_counter()-start,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9))
        print(records[-1],flush=True);del c,support,w,grad,logits,scores;torch.cuda.empty_cache()
    assert state_sha(s.model)==MODEL_HASH
    dump(HERE/'preflight.json',dict(passed=True,records=records,torch=torch.__version__,pyg=__import__('torch_geometric').__version__,device=torch.cuda.get_device_name(0),
         source_hashes={str(p):sha(p) for p in [HERE/'bridge.py',Path(__file__),REPAIR/'transaction_adapter_v2.py']},model_unchanged=True))
if __name__=='__main__':main()

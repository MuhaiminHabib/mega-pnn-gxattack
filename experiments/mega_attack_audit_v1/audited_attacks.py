"""Audited MEGA-PNA transaction adaptations; fixed one-confirmation protocol.

GXAttack optimizes the frozen PGExplainer objective; its edits transfer to
GNNExplainer. Deduction optimizes the target explainer loss separately for
each explainer. Original top-five transactions are protected for deduction.
"""
from repair_core import *
from audit import exact_project

CONFIG=dict(seed=101,budget=10,top_k=5,candidate_additions=256,
            gx_epochs=100,gx_restarts=3,gx_step=.1,gx_beta=.1,gx_entropy=.0005,
            gx_rounding_draws=256,gx_min_edits=5,
            deduction_epochs=20,deduction_restarts=1,degree_threshold=.000157,
            explainer_seed=20261017,confirmation_seed_offset=100000,
            gnn_epochs=1500,gnn_lr=.01)

def project(p,budget):
    return exact_project(p,budget)

def weights(base,flips):
    w=base.clone()
    if flips:w[flips]=1-w[flips]
    return w

def likelihood(s,c,base,flips):
    new=s.degree.copy();ei=c['nodes'][c['edge_index'][:,flips]].cpu().numpy()
    for j,i in enumerate(flips):np.add.at(new,ei[:,j],1 if base[i]==0 else -1)
    def ll(d):
        d=d[d>=5];n=len(d);a=1+n/np.log(d/4.5).sum()
        return n*np.log(a)+n*a*np.log(5)-(a+1)*np.log(d).sum()
    return float(-2*ll(np.concatenate([s.degree,new]))+2*ll(s.degree)+2*ll(new))

def deduction_ranking(s,clean_c,support,base,protected,kind,method,seed,progress):
    """Same published deduction objective; TX inputs and identity-based GNN init."""
    seed_all(seed);m=int(base.sum());device=base.device
    indices=torch.arange(m,device=device) if kind=='D' else torch.tensor(protected+list(range(m,len(base))),device=device)
    if len(indices)==0:return [],[]
    fixed=torch.isin(indices,torch.tensor(protected,device=device))
    if method=='pgexplainer':
        with torch.no_grad():
            if kind=='A':
                auxiliary=torch.zeros_like(base);auxiliary[indices]=1
                _,z=s.predict(support,auxiliary,embedding=True)
            else:
                _,z=s.predict(clean_c,embedding=True)
            inputs=s.pg_inputs(z,support)[indices].detach()
        network=nn.Sequential(nn.Linear(inputs.size(1),64),nn.ReLU(),nn.Linear(64,1)).to(device)
        params=list(network.parameters());opt=torch.optim.Adam(params,lr=.003);size=CFG['pg_edge_size']
        def raw():return network(inputs).flatten()
    else:
        initial=id_normal(support['tx_rows'][indices].cpu().numpy(),seed,len(clean_c['x']))+1.
        parameter=nn.Parameter(torch.from_numpy(initial).to(device))
        params=[parameter];opt=torch.optim.Adam(params,lr=.01);size=CFG['gnn_edge_size']
        def raw():return parameter
    history=[]
    for epoch in range(CONFIG['deduction_epochs']):
        opt.zero_grad();noise=torch.rand(len(indices),device=device).clamp(.0001,.9999)
        temperature=5*((2/5)**(epoch/30));total=0.
        for j,beta in enumerate([.001,.7,.7996666667,.8993333333,.999]):
            r=raw();mask=((r+noise.log()-(1-noise).log())/temperature).sigmoid() if method=='pgexplainer' else r.sigmoid()
            effective=torch.where(fixed,torch.full_like(mask,beta),mask)
            w=base.clone() if kind=='D' else torch.zeros_like(base)
            w=w.index_copy(0,indices,effective)
            logits=s.predict(support,w)[support['target']]
            q=effective.clamp(1e-6,1-1e-6)
            loss=F.cross_entropy(logits[None],torch.ones(1,dtype=torch.long,device=device))+size*q.sum()+(-(q*q.log()+(1-q)*(1-q).log())).mean()
            coefficient=(-4 if j==0 else 1)*(1 if kind=='D' else -1)
            assert torch.isfinite(loss);(loss*coefficient).backward();total+=float(loss.detach())*coefficient
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        opt.step();history.append(total)
        progress(stage='deduction_optimize',explainer=method,kind=kind,epoch=epoch+1)
    with torch.no_grad():scores=raw().sigmoid().cpu().numpy()
    ranking=sorted([(int(i),float(v)) for i,v in zip(indices.cpu(),scores) if int(i) not in protected],key=lambda x:(-x[1],int(support['tx_rows'][x[0]])))
    return [i for i,v in ranking],history


def deduction(s,clean_c,support,base,clean,method,explain,progress):
    target=support['global_node'];m=int(base.sum());ids=support['tx_rows'].cpu().tolist();mapping={i:j for j,i in enumerate(ids)}
    protected=[mapping[i] for i in clean['top5']]
    proposals=[];traces=[]
    for restart in range(CONFIG['deduction_restarts']):
        seed=CONFIG['seed']+target+restart*1000
        deletion,dh=deduction_ranking(s,clean_c,support,base,protected,'D',method,seed,progress)
        addition,ah=deduction_ranking(s,clean_c,support,base,protected,'A',method,seed+1,progress)
        traces.append(dict(restart=restart,deletion_loss=dh,addition_loss=ah))
        for nadd in range(CONFIG['budget']+1):
            ndel=CONFIG['budget']-nadd
            if len(deletion)>=ndel and len(addition)>=nadd:proposals.append(deletion[:ndel]+addition[:nadd])
    def select(candidates,arm):
        best=None;checks=[]
        for j,flips in enumerate(candidates):
            progress(stage='deduction_select',explainer=method,arm=arm,proposal=j+1,total=len(candidates))
            ratio=likelihood(s,support,base,flips);w=weights(base,flips)
            with torch.no_grad():p=float(s.predict(support,w)[support['target']].softmax(0)[1])
            ok=ratio<CONFIG['degree_threshold'] and p>.5
            entry=dict(flips=flips,p=p,degree_ratio=ratio,valid=ok)
            if ok:
                after=explain(method,flips,CONFIG['explainer_seed']+target)
                entry['change']=change(clean,after)
                if best is None or entry['change']['top5_changed_fraction']>best['change']['top5_changed_fraction']:best=entry
            checks.append(entry)
        return best,checks
    best,checks=select(proposals,'attack')
    rng=random.Random(CONFIG['seed']+target+200003);allowed=[i for i in range(m) if i not in protected]
    additions=list(range(m,len(base)));ndel=sum(i<m for i in best['flips']) if best else min(len(allowed),CONFIG['budget'])
    nadd=CONFIG['budget']-ndel
    # Ordinary random is one prespecified draw, not the best draw in a search.
    ordinary_flips=sorted(rng.sample(allowed,ndel)+rng.sample(additions,nadd))
    _,ordinary_checks=select([ordinary_flips],'ordinary_random')
    # Match every attack candidate's composition and count before evaluation.
    random_proposals=[];seen=set()
    for candidate in proposals:
        d=sum(i<m for i in candidate);a=len(candidate)-d
        while True:
            flips=sorted(rng.sample(allowed,d)+rng.sample(additions,a))
            if tuple(flips) not in seen:break
        seen.add(tuple(flips));random_proposals.append(flips)
    random_best,random_checks=select(random_proposals,'random')
    assert len(checks)==len(random_checks)
    return dict(attack=best,random=random_best,ordinary=ordinary_checks[0],protected_ids=clean['top5'],traces=traces,attack_proposals=checks,random_proposals=random_checks,
                random_matching='one random candidate per evaluated attack candidate, matching its composition; equal distinct evaluated counts; same objective; validity not resampled',ordinary_matching='single prespecified draw matched to attack winner composition, separate from search')

def gx(s,support,base,progress):
    target=support['global_node'];m=int(base.sum())
    with torch.no_grad():_,initial=s.pg_scores(support,base);initial=initial*base
    best=None;history=[];seen=set();checks=[]
    def score(flips):
        w=weights(base,flips)
        with torch.no_grad():
            logits,v=s.pg_scores(support,w);v=v*w;p=float(logits[support['target']].softmax(0)[1])
            objective=float(F.log_softmax(logits[support['target']],0)[1]+CONFIG['gx_beta']*(1-F.cosine_similarity(initial[None],v[None])[0]))
        return dict(flips=flips,p=p,objective=objective,valid=p>.5)
    for restart in range(CONFIG['gx_restarts']):
        seed_all(CONFIG['seed']+target+60000+restart)
        p=torch.zeros_like(base) if restart==0 else project(torch.rand_like(base),5)
        p.requires_grad_(True);trace=[]
        for epoch in range(CONFIG['gx_epochs']):
            # At exactly0/1 the fixed PNA adapter has clamp-induced endpoint
            # gradients that fail one-sided finite differences. Evaluate the
            # unchanged objective's gradient at a nearby interior point only.
            # This is an explicit numerical transaction adaptation, not a new loss.
            interior=p.detach().clamp(1e-4,1-1e-4).requires_grad_(True)
            w=base+(1-2*base)*interior
            logits,values=s.pg_scores(support,w);values=values*w
            pred=F.log_softmax(logits[support['target']],0)[1]
            distance=1-F.cosine_similarity(initial[None],values[None])[0]
            q=interior*.99+.005;entropy=-(q*q.log()+(1-q)*(1-q).log()).mean()
            obj=pred+CONFIG['gx_beta']*distance-CONFIG['gx_entropy']*entropy
            grad=torch.autograd.grad(obj,interior)[0];assert torch.isfinite(grad).all()
            with torch.no_grad():p.copy_(project(p+CONFIG['gx_step']*grad,CONFIG['budget']))
            if epoch%10==0:trace.append(dict(epoch=epoch+1,objective=float(obj),expected_edits=float(p.sum()),gradient_norm=float(grad.norm())))
            progress(stage='gx_optimize',restart=restart+1,epoch=epoch+1,epochs=CONFIG['gx_epochs'])
        probabilities=p.detach();history.append(trace)
        seed_all(CONFIG['seed']+target+7000+restart*101)
        for j in range(CONFIG['gx_rounding_draws']):
            flips=torch.bernoulli(probabilities).nonzero().flatten().tolist()
            if not CONFIG['gx_min_edits']<=len(flips)<=CONFIG['budget'] or tuple(flips) in seen:continue
            seen.add(tuple(flips));entry=score(flips);checks.append(entry)
            if entry['valid'] and (best is None or entry['objective']>best['objective']):best=entry
            if j%16==0:progress(stage='gx_rounding',restart=restart+1,draw=j+1)
    rng=random.Random(CONFIG['seed']+target+300007)
    ndel=sum(i<m for i in best['flips']) if best else min(CONFIG['budget'],m)
    nadd=(len(best['flips'])-ndel) if best else 0
    ordinary=score(sorted(rng.sample(range(m),ndel)+rng.sample(range(m,len(base)),nadd)))
    random_best=None;random_checks=[];random_seen=set()
    for j,candidate in enumerate(checks):
        d=sum(i<m for i in candidate['flips']);a=len(candidate['flips'])-d
        while True:
            flips=sorted(rng.sample(range(m),d)+rng.sample(range(m,len(base)),a))
            if tuple(flips) not in random_seen:break
        random_seen.add(tuple(flips))
        entry=score(flips);random_checks.append(entry)
        if entry['valid'] and (random_best is None or entry['objective']>random_best['objective']):random_best=entry
        if j%16==0:progress(stage='gx_random_control',draw=j+1,total=len(checks))
    assert len(checks)==len(random_checks)
    return dict(attack=best,random=random_best,ordinary=ordinary,history=history,attack_proposals=checks,random_proposals=random_checks,
                random_matching='one random candidate per evaluated attack candidate, matching its composition; equal distinct evaluated counts; same objective; validity not resampled',ordinary_matching='single prespecified draw matched to attack winner composition, separate from search')

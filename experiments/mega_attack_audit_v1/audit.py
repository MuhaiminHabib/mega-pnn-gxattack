"""Read-only model diagnostics and executable checks against committed authors."""
import ast, json, sys, types, pickle, hashlib, msvcrt
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'mega_priority_repair'))
import repair_core as core
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

def function(file,name,namespace):
    tree=ast.parse((ROOT/'reference'/file).read_text(encoding='utf-8-sig'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),file,'exec'),namespace)
    return namespace[name]

def exact_project(p,budget):
    # Project the un-clipped vector; clipping first changes the Euclidean solution.
    if float(p.clamp(0,1).sum())<=budget:return p.clamp(0,1)
    lo=float((p-1).min());hi=float(p.max())
    for _ in range(60):
        mid=(lo+hi)/2
        if float((p-mid).clamp(0,1).sum())>budget:lo=mid
        else:hi=mid
    return (p-hi).clamp(0,1)

class Record: pass
class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('graphxai'):
            return Record
        return super().find_class(module,name)

def reference():
    torch.set_num_threads(2)
    src=Path(r'C:\habib dissertation\Projects\GXAttack')
    ckpt=src/'reproduction/runs/dataset_Syn1_seed0/gcn.pt'
    dataset=src/'exp_results/dataset_Syn1.pth'
    # Deserialize only stored data; do not import legacy training/debug modules.
    shim=types.ModuleType('data_only_pickle');shim.Unpickler=DataOnlyUnpickler;shim.load=pickle.load
    data=torch.load(dataset,map_location='cpu',weights_only=False,pickle_module=shim).graph
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__();self.gcn1=GCNConv(10,32);self.gcn2=GCNConv(32,2)
        def forward(self,x,ei,w):return self.gcn2(self.gcn1(x,ei,w).relu(),ei,w)
    model=GCN();model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));model.double().eval()
    for p in model.parameters():p.requires_grad_(False)
    x=data.x.double();n=len(x)
    pairs=torch.tensor([(a,b) for a in range(n) for b in range(a+1,n)]).T
    ei=torch.cat([pairs,pairs.flip(0)],1)
    present=set(map(tuple,data.edge_index.T.tolist()))
    base=torch.tensor([float(tuple(p) in present or tuple(p[::-1]) in present) for p in pairs.T.tolist()],dtype=torch.double)
    clean=model(x,ei,base.repeat(2));target=int(clean.argmax(1)[0]);node=0
    ns={'torch':torch,'self':types.SimpleNamespace(coeff_ent=.0005)}
    l1=function('GXAttacker.py','loss_fn1',ns);l2=function('GXAttacker.py','loss_fn2_cos',ns)
    function('GXAttacker.py','_bisection',ns);proj=function('GXAttacker.py','_project',ns)
    pg_ns={'torch':torch}
    pg_loss=function('PGAttacker.py','pg_loss',pg_ns)
    dloss=function('PGAttacker.py','D_loss',pg_ns);aloss=function('PGAttacker.py','A_loss',pg_ns)
    proxy=types.SimpleNamespace(N=4);proxy.pg_loss=types.MethodType(pg_loss,proxy)
    # A fixed differentiable edge-scoring probe isolates GX's attack math.
    # It is NOT a trained PGExplainer or a paper attack-success benchmark.
    def outputs(p):
        w=base+(1-2*base)*p;logits=model(x,ei,w.repeat(2))
        scores=(logits[pairs[0],target]+logits[pairs[1],target]).sigmoid()*w
        return logits,scores
    with torch.no_grad():_,initial=outputs(torch.zeros_like(base))
    def obj(p,author):
        logits,scores=outputs(p);prob=logits[node].softmax(0)
        if author:
            return -l1(prob,target,p)+.1*l2(n,initial.repeat(2),scores.repeat(2),ei,ei)[0]
        q=.99*p+.005;ent=-(q*q.log()+(1-q)*(1-q).log()).mean()
        return (prob[target]+1e-6).log()+.1*(1-F.cosine_similarity(initial[None],scores[None])[0])-.0005*ent
    a=torch.full_like(base,.003);b=a.clone();steps=[]
    for step in range(5):
        a.requires_grad_();b.requires_grad_();av=obj(a,True);bv=obj(b,False)
        ag=torch.autograd.grad(av,a)[0];bg=torch.autograd.grad(bv,b)[0]
        steps.append(dict(step=step,value_error=abs(float(av-bv)),gradient_max_error=float((ag-bg).abs().max())))
        a=proj(10,a.detach()+.1*ag,0).detach();b=exact_project(b.detach()+.1*bg,10)
    assert max(r['gradient_max_error'] for r in steps)<1e-8
    assert torch.allclose(a,b,atol=2e-5,rtol=0)
    discrepancy=torch.tensor([.9,2.,.1],dtype=torch.double)
    old=core.original_attacks.project(discrepancy,1);correct=exact_project(discrepancy,1);author=proj(1,discrepancy,0)
    assert torch.allclose(correct,author,atol=1e-5,rtol=0) and not torch.allclose(old,correct,atol=.01)
    deduction=[]
    # Original graph; the same raw mask drives all five protected-edge settings.
    edge_index=data.edge_index;weights=torch.ones(edge_index.shape[1],dtype=torch.double)
    for kind,author_fn,sign in [('D',dloss,1),('A',aloss,-1)]:
        raw=torch.linspace(-.5,.5,len(weights),dtype=torch.double,requires_grad=True)
        masks=[];preds=[]
        fixed=torch.arange(len(raw))<5
        for beta in [.001,.7,.7996666667,.8993333333,.999]:
            q=torch.where(fixed,torch.full_like(raw,beta),raw.sigmoid())
            masks.append(q);preds.append(model(x,edge_index,q)[node:node+1])
        label=torch.tensor([target]);ref=author_fn(proxy,masks,preds,label,raw,(.0003,1.))
        vals=[F.cross_entropy(p,label)+.0003*m.sum()-(m*m.log()+(1-m)*(1-m).log()).mean() for m,p in zip(masks,preds)]
        port=sign*(-4*vals[0]+sum(vals[1:]));rg=torch.autograd.grad(ref,raw,retain_graph=True)[0];pg=torch.autograd.grad(port,raw)[0]
        assert torch.allclose(rg,pg,atol=1e-10,rtol=1e-8)
        deduction.append(dict(kind=kind,value_error=abs(float(ref-port)),gradient_max_error=float((rg-pg).abs().max()),ranking_equal=bool(torch.equal(rg.argsort(),pg.argsort()))))
    # The GNN author's signed sum is also checked directly on the same losses.
    gn={};gd=function('GNN_Attacker.py','dedu_loss',gn)
    gself=types.SimpleNamespace(graph_mode=True,label=0)
    gself.loss=lambda mask,pred,pred_label,node_idx:pred+0
    values=[torch.tensor(float(i+1),requires_grad=True) for i in range(5)]
    gr=gd(gself,[None]*5,values,0,0,'hot',[.7,.8,.9,.999]);gp=-4*values[0]+sum(values[1:])
    assert float(gr)==float(gp)
    report=dict(passed=True,reference_case=dict(dataset=str(dataset),dataset_sha256=core.sha(dataset),checkpoint=str(ckpt),checkpoint_sha256=core.sha(ckpt),nodes=n,directed_edges=int(data.edge_index.shape[1]),node=node,prediction=target),gx_five_steps=steps,gx_final_parameter_max_error=float((a-b).abs().max()),deduction_pg=deduction,gnn_deduction_sign_check=True,
        projection_counterexample=dict(input=discrepancy.tolist(),budget=1,old=old.tolist(),correct=correct.tolist(),author=author.tolist()),
        scope='Executable loss/gradient/update parity on saved Syn1 GCN; fixed edge-scoring probe, not end-to-end author PG training or paper success reproduction.',
        source_notes=['GX author code uses sigmoid scores, all-pair dense symmetric graphs, stochastic PG sampling, hardcoded step .01 and budget15. Existing transaction adaptation uses deterministic frozen PG and sparse directed transactions; retain explicit adaptation label.',
        'PG deduction author addition inputs are computed on its complementary graph. Existing port incorrectly reused clean-graph embeddings for that auxiliary stage.',
        'PG author addition target is predicted on complementary graph; our retained illicit target follows the prediction-preserving research question and is an explicit adaptation.',
        'GNN author _masked_adj can set protected sigmoid(0)+beta above1; do not copy this legacy artifact into a Bernoulli transaction mask.',
        'Reference regularization constants can differ from PyG; signed-sum regularizer gradients cancel. Keep selected explainer loss constants.'])
    core.dump(ROOT/'reference_audit.json',report)
    print('REFERENCE AUDIT PASSED',flush=True)

def diagnostics():
    s=core.RepairStudy();core.load_pg(s);node=588;c,support,base,_=s.context(node)
    with torch.no_grad():
        logits,z=s.predict(support,base,True);raw=s.pg.mlp(s.pg_inputs(z,support)).flatten();initial=raw.sigmoid()*base
    top=torch.argsort(raw[:int(base.sum())],descending=True)[:5]
    def terms(p):
        w=base+(1-2*base)*p;logits,scores=s.pg_scores(support,w)
        pred=F.log_softmax(logits[support['target']],0)[1]
        dist=1-F.cosine_similarity(initial[None],(scores*w)[None])[0]
        q=.99*p+.005;ent=-(q*q.log()+(1-q)*(1-q).log()).mean()
        return pred,dist,ent
    def objective(p):
        a,b,e=terms(p);return a+.1*b-.0005*e
    checks=[];all_grad=None
    for at in [0.,.05]:
        p=torch.full_like(base,at,requires_grad=True);pred,dist,ent=terms(p)
        gradients=[torch.autograd.grad(t,p,retain_graph=True)[0] for t in [pred,dist,ent]]
        g=gradients[0]+.1*gradients[1]-.0005*gradients[2]
        if at==0:all_grad=g.detach()
        index=int(g.abs().argmax());eps=.001
        plus=p.detach().clone();minus=p.detach().clone();plus[index]+=eps
        if at>0:minus[index]-=eps
        with torch.no_grad():fd=float((objective(plus)-objective(minus))/(eps*(2 if at>0 else 1)))
        checks.append(dict(at=at,index=index,automatic=float(g[index]),finite_difference=fd,absolute_error=abs(float(g[index])-fd),prediction_grad_norm=float(gradients[0].norm()),distance_grad_norm=float(gradients[1].norm()),entropy_grad_norm=float(gradients[2].norm()),objective=float(pred+.1*dist-.0005*ent)))
        assert torch.isfinite(g).all()
    # Predetermined top8 gradient plus8 seeded random coordinates, no attack optimization.
    rng=np.random.default_rng(20261016);indices=sorted(set(all_grad.argsort(descending=True)[:8].cpu().tolist()+rng.choice(len(base),8,replace=False).tolist()))
    with torch.no_grad():clean_obj=float(objective(torch.zeros_like(base)))
    binary=[]
    for i in indices:
        flip=torch.zeros_like(base);flip[i]=1
        with torch.no_grad():v=float(objective(flip));prob=float(s.predict(support,core.original_attacks.weights(base,[i]))[support['target']].softmax(0)[1])
        binary.append(dict(index=i,operation='delete' if base[i] else 'add',gradient=float(all_grad[i]),objective_gain=v-clean_obj,p=prob))
    protected=top.cpu().tolist();ix=torch.tensor(protected+list(range(int(base.sum()),len(base))),device=base.device)
    w=torch.zeros_like(base);w[ix]=1
    with torch.no_grad():
        _,zc=s.predict(c,embedding=True);la,za=s.predict(support,w,True)
        old=s.pg_inputs(zc,support)[ix];correct=s.pg_inputs(za,support)[ix]
    report=dict(node=node,split='validation',source='previous pilot diagnostic only',sigmoid_top5=raw[top].sigmoid().tolist(),raw_top5=raw[top].tolist(),original_edges=int(base.sum()),saturated_ones=int((raw[:int(base.sum())].sigmoid()==1).sum()),actual_gx_gradient_checks=checks,binary_single_edit_checks=binary,
                pg_addition_embedding_max_difference=float((old-correct).abs().max()),pg_addition_graph_illicit_probability=float(la[support['target']].softmax(0)[1]),
                frozen_detector_state=core.state_sha(s.model),passed_finite=True)
    assert core.state_sha(s.model)==core.MODEL_HASH
    core.dump(ROOT/'mega_diagnostics.json',report);print('MEGA DIAGNOSTICS SAVED',flush=True)

if __name__=='__main__':
    lock=open(ROOT.parent/'mega_attack_eval/worker.lock','a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    assert json.loads((ROOT.parent/'mega_attack_eval/results_v1/status.json').read_text())['state']=='paused'
    reference();diagnostics()

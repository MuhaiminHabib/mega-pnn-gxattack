"""Sixteen fixed updates per copy, using only labels of training targets."""
import copy,sys,time,traceback
from common import *
def main(arm):
    p,lock,guard,pc,attacks,s=start(f'Retraining {arm}')
    assert arm in ['clean','random','adversarial'];torch=pc.torch;F=pc.F
    pc.seed_all(p['training_seed']);baseline=pc.state_sha(s.model)
    model=pc.make_model(s.graph).to(pc.DEVICE)
    model.load_state_dict(s.model.state_dict());model.eval()
    for par in model.parameters():par.requires_grad_(True)
    guard._hooks.append(model.register_forward_pre_hook(guard.tick))
    original_buffers={k:v.clone() for k,v in model.named_buffers()}
    opt=torch.optim.Adam(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    nodes=p['cohort']['train_nodes'];assert len(nodes)==8
    contexts={};edited={};valid_banks=0
    for node in nodes:
        assert bool(s.masks['train'][node]) and not bool(s.masks['validation'][node]) and not bool(s.masks['test'][node])
        if node in p['cohort']['attack_training_nodes']:
            bank=read(ROOT/f'bank/{node}.json');saved=torch.load(ROOT/f'bank/{node}_contexts.pt',map_location='cpu',weights_only=False)
            c=pc.move_context(saved['clean']);d=c
            if bank['paired_training_edits_valid']:
                valid_banks+=1
                if arm!='clean':
                    support=pc.move_context(saved['support']);base=saved['base'].to(pc.DEVICE)
                    flips=bank['ordinary' if arm=='random' else 'attack']['flips'];d=pc.subset(support,attacks.weights(base,flips).bool())
        else:c=pc.move_context(pc.build_context(s.graph,node));d=c
        contexts[node]=c;edited[node]=d
    assert valid_banks>0,'No valid training edit pair; do not claim adversarial fine-tuning'
    checkpoint=ROOT/f'checkpoints/train_{arm}.pt';logs=[];start_step=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=pc.DEVICE,weights_only=False)
        assert saved['protocol_sha256']==sha(ROOT/'protocol.json') and saved['arm']==arm
        model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer']);logs=saved['logs'];start_step=saved['step']
    for step in range(start_step,p['updates_per_arm']):
        node=nodes[step%len(nodes)];c=contexts[node];d=edited[node]
        target=s.graph['node'].y[node].to(pc.DEVICE).view(1)
        opt.zero_grad(set_to_none=True)
        clean_logits=pc.forward(model,pc.official(c,pc.DEVICE))[c['target']].view(1,-1)
        aug_logits=pc.forward(model,pc.official(d,pc.DEVICE))[d['target']].view(1,-1)
        ce_clean=F.cross_entropy(clean_logits,target);ce_aug=F.cross_entropy(aug_logits,target)
        a=clean_logits.softmax(-1);b=aug_logits.softmax(-1);m=(a+b)/2
        js=.5*((a*(a.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum()+(b*(b.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum())
        loss=.5*(ce_clean+ce_aug)+js;assert torch.isfinite(loss)
        loss.backward();assert all(par.grad is None or torch.isfinite(par.grad).all() for par in model.parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']));opt.step()
        assert all(torch.equal(original_buffers[k],v) for k,v in model.named_buffers()),'BN buffers changed'
        logs.append(dict(step=step+1,node=node,label=int(target.item()),clean_ce=float(ce_clean.detach()),augmented_ce=float(ce_aug.detach()),js=float(js.detach()),gradient_norm=norm))
        save_torch(torch,checkpoint,dict(arm=arm,step=step+1,state_dict=model.state_dict(),optimizer=opt.state_dict(),logs=logs,protocol_sha256=sha(ROOT/'protocol.json')))
        status(stage='training',arm=arm,completed_updates=step+1,total_updates=p['updates_per_arm'])
    assert len(logs)==16 and pc.state_sha(model)!=baseline and pc.state_sha(s.model)==baseline
    save_torch(torch,ROOT/f'models/{arm}.pt',dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        state_sha256=pc.state_sha(model),baseline_state_sha256=baseline,protocol_sha256=sha(ROOT/'protocol.json'),arm=arm))
    write(ROOT/f'models/{arm}.json',dict(arm=arm,updates=16,training_nodes=nodes,valid_augmentation_targets=valid_banks,
        baseline_unchanged=True,bn_buffers_unchanged=True,state_sha256=pc.state_sha(model),checkpoint_sha256=sha(ROOT/f'models/{arm}.pt'),logs=logs))
    validate();guard.close();print('TRAIN COMPLETE',arm,flush=True)
if __name__=='__main__':
    try:main(sys.argv[1])
    except BaseException:write(ROOT/'failure.json',dict(stage='train',traceback=traceback.format_exc()));raise

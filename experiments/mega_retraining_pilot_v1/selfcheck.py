"""Native-loop and optimizer-resume check on a CPU toy; no experimental outcomes."""
import ast
from common import *
from traced import algorithm_class
def main():
    p=validate();pc,_=science();torch,np=pc.torch,pc.np;torch.set_num_threads(2)
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf8'))
    from torch_geometric.nn import GCNConv
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.conv=GCNConv(2,2)
        def forward(self,x,edge_index):return self.conv(x,edge_index)
    class Native(pc.IdentityGNN):
        def _clean_model(self,model):
            self.saved_raw=self.edge_mask.detach().clone();self.saved_hard=self.hard_edge_mask.detach().clone()
            return super()._clean_model(model)
    pc.seed_all(12);model=Tiny().eval()
    for par in model.parameters():par.requires_grad_(False)
    x=torch.tensor([[1.,0.],[.3,.8],[.1,.4],[.9,.6]]);edges=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    ids=np.arange(6);labels=torch.ones(4,dtype=torch.long)
    def fit(a):
        a.connect(pc.ExplainerConfig(explanation_type='model',edge_mask_type='object',node_mask_type=None),pc.ModelConfig(mode='multiclass_classification',task_level='node',return_type='raw'))
        a(model,x,edges,target=labels,index=1);return a
    native=Native(ids,77);native.epochs=12;fit(native)
    Traced=algorithm_class(pc);a=fit(Traced(ids,77,12,interval=2))
    assert np.array_equal(a.saved_raw,native.saved_raw.numpy()) and a.trace==native.trace
    checkpoint=ROOT/'checkpoints/selfcheck_toy.pt';assert not checkpoint.exists()
    fit(Traced(ids,77,4,checkpoint=checkpoint,interval=2));b=fit(Traced(ids,77,12,checkpoint=checkpoint,interval=2))
    assert np.array_equal(a.saved_raw,b.saved_raw) and a.trace==b.trace
    assert set(p['cohort']['attack_training_nodes'])<=set(p['cohort']['train_nodes'])
    assert not set(p['cohort']['train_nodes'])&set(p['cohort']['validation_nodes'])
    assert len(p['cohort']['train_nodes'])*p['epochs']==p['updates_per_arm']==16
    write(ROOT/'selfcheck.json',dict(passed=True,native_loop_bitwise_equal=True,resume_bitwise_equal=True,
        train_validation_disjoint=True,cpu_toy_only=True))
    print('PASS native loop, optimizer resumption, split and update-count checks',flush=True)
if __name__=='__main__':main()

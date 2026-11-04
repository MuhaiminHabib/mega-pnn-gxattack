"""Freeze the user-authorized ten-node protocol before any new outcomes."""
import hashlib
import json
from pathlib import Path
from resource_guard import POLICY,write_json
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parent=ROOT.parent/'mega_attack_audit_v1/results/protocol.json'
    previous=read(parent);hashes=previous['source_hashes'].copy()
    assert all(sha(p)==h for p,h in hashes.items()),'Parent frozen input changed'
    assert read(ROOT/'infrastructure_verification.json')['passed']
    files=list(ROOT.glob('*.py'))+[ROOT/x for x in ['README.txt','resume.ps1','cohort.json','infrastructure_verification.json']]
    hashes.update({str(p.resolve()):sha(p) for p in files});hashes[str(parent.resolve())]=sha(parent)
    cohort=read(ROOT/'cohort.json')
    p=dict(previous)
    p.update(cohorts=dict(validation_pilot=cohort['validation_nodes']),expected=40,
        source_hashes=hashes,parent_protocol_sha256=sha(parent),resource_policy=POLICY,
        cohort_selection=cohort,stopping='Ten validation nodes then verify/report. No training/tuning/defenses/full-run resumption.',
        caveats=['Ten validation accounts; earlier known development/confirmation targets excluded.',
                 'Separate from preceding four-node audit; no favorable-outcome selection or size exclusions.',
                 'Author function audit is not end-to-end paper reproduction; transaction adaptations retained.',
                 'Equal discrete search counts/compositions do not mean equal total compute.',
                 'No-valid arms contribute zero prediction-preserving displacement, with validity denominators reported.',
                 'Account-paired bootstrap; top5 primary/top25 secondary. Displacement is not F1 loss.',
                 'Resource holds retain completed work; an unfinished fit may restart. No node substitution after a memory failure.'])
    target=ROOT/'results/protocol.json'
    if target.exists():assert read(target)==p,'Frozen ten-node protocol differs'
    else:write_json(target,p)
    write_json(ROOT/'results/status.json',dict(state='queued_after_current_pilot',completed=0,expected=40,nodes=10))
    print('FROZEN',len(hashes),'files; 10 nodes, 40 combinations; protocol',sha(target))


if __name__=='__main__':main()

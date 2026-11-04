"""Serial continuation after the existing GNN convergence worker exits."""
import sys,json,subprocess,traceback,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(name,obj):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def main():
    assert read(ROOT.parent/'mega_attack_eval/results_v1/status.json')['state']=='paused'
    plan=read(ROOT/'plan.json')
    for p,h in plan['source_manifest'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert (ROOT/'gnn_convergence/summary.json').exists(),'Initial GNN convergence stage did not finish; inspect worker/failure before resuming.'
    steps=[('fit_pg.py','pg_normalized/summary.json'),('confirm.py','confirmation/summary.json'),('attack_pilot.py','attack_summary.json'),('report.py','independent_verification.json')]
    for script,completion in steps:
        if (ROOT/completion).exists():continue
        write('pipeline_status.json',dict(state='running',stage=script,updated=time.time()))
        with (ROOT/(script+'.log')).open('a',encoding='utf8') as log:
            result=subprocess.run([sys.executable,'-B','-u',str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT.parents[1]))
        assert result.returncode==0,f'{script} failed; inspect {script}.log'
        assert (ROOT/completion).exists(),f'{script} exited without its completion artifact'
    write('pipeline_status.json',dict(state='complete',updated=time.time()))
if __name__=='__main__':
    try:main()
    except Exception:write('pipeline_failure.json',dict(traceback=traceback.format_exc()));raise

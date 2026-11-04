"""Bounded infrastructure checks: no graph loading, GPU work or model fitting."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from resource_guard import POLICY, hold_reason, write_json

ROOT=Path(__file__).resolve().parent


def check_unpaced_monitoring():
    from types import SimpleNamespace
    from unittest.mock import patch
    import resource_guard as rg
    calls=[]
    guard=rg.Guard.__new__(rg.Guard)
    guard.ticks=0;guard.window=0.;guard.sleep_seconds=0.
    guard.torch=SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True, synchronize=lambda:calls.append('sync')))
    guard.check=lambda:calls.append('check')
    with patch.object(rg.time,'perf_counter',return_value=1.), patch.object(rg.time,'sleep',side_effect=lambda delay:calls.append(('sleep',delay))):
        guard.tick()
    assert calls==['check'] and guard.ticks==1 and guard.sleep_seconds==0.
    with patch.object(rg.time,'perf_counter',return_value=1.25), patch.object(rg.time,'sleep',side_effect=AssertionError('Unexpected sleep')):
        guard.tick()
    assert calls==['check'] and guard.ticks==2
    # The resource check can still stop a due forward before it runs.
    class ResourceStop(Exception):pass
    def stop():raise ResourceStop()
    guard.check=stop
    with patch.object(rg.time,'perf_counter',return_value=2.):
        try:guard.tick()
        except ResourceStop:pass
        else:raise AssertionError('Resource stop suppressed')
    # Preserve the existing synchronization and delay if pacing is ever enabled.
    calls.clear();guard.window=0.;guard.check=lambda:calls.append('check')
    with patch.dict(rg.POLICY,{'gpu_active_fraction':.4}), patch.object(rg.time,'perf_counter',return_value=1.), patch.object(rg.time,'sleep',side_effect=lambda delay:calls.append(('sleep',delay))):
        guard.tick()
    assert calls==['sync','check',('sleep',1.5)] and guard.sleep_seconds==1.5


def main():
    check_unpaced_monitoring()
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    assert hold_reason({'used_bytes':29_400_000_000},65) is None
    assert hold_reason({'used_bytes':29_500_000_000},65)=='system_ram_ceiling'
    assert hold_reason({'used_bytes':10_000_000_000},75)=='gpu_temperature'
    # Prove a small real Windows job limit rejects an over-limit allocation;
    # no deliberate pressure on the machine's physical RAM.
    code="""from resource_guard import cap_own_process
h,n=cap_own_process(64*1024**2+1536)
assert n==64*1024**2
try: a=bytearray(128*1024**2)
except MemoryError: print('CAP_ENFORCED')
else: raise AssertionError('Allocation cap was not enforced')
"""
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0 and 'CAP_ENFORCED' in result.stdout,(result.stdout,result.stderr)
    probe="from resource_guard import cap_own_process,POLICY; import mmap; h,n=cap_own_process(POLICY['process_commit_limit_bytes']); assert n==(POLICY['process_commit_limit_bytes']//mmap.PAGESIZE)*mmap.PAGESIZE; print(n)"
    actual=subprocess.run([sys.executable,'-B','-c',probe],cwd=ROOT,capture_output=True,text=True)
    assert actual.returncode==0,(actual.stdout,actual.stderr)
    # Test the preflight comparison without importing the GPU runner.
    tree=ast.parse((ROOT/'run.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_pg_repeat')
    scope={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'check_pg_repeat','exec'),scope)
    check=scope['check_pg_repeat']
    evidence=json.loads((ROOT/'recovery_pg_preflight_roundoff/diagnosis.json').read_text())
    a=dict(top5=evidence['top5_first'],top5_scores=evidence['scores_first'])
    b=dict(top5=evidence['top5_second'],top5_scores=evidence['scores_second'])
    assert check(a,b)==evidence['max_absolute_score_error']
    for bad in [dict(top5=[-1]+b['top5'][1:],top5_scores=b['top5_scores']),dict(top5=b['top5'],top5_scores=[b['top5_scores'][0]+.01]+b['top5_scores'][1:])]:
        try:check(a,bad)
        except AssertionError:pass
        else:raise AssertionError('Changed ranking or material score difference was accepted')
    c=json.loads((ROOT/'cohort.json').read_text())
    assert len(c['validation_nodes'])==len(set(c['validation_nodes']))==10
    assert not set(c['validation_nodes'])&set(c['excluded_eligible_nodes'])
    assert all(r['split']=='validation' and r['correctly_detected_illicit'] for r in c['selected_census_rows'])
    # Only the cohort count and a per-process account selector differ in cases.
    original=(ROOT.parent/'mega_attack_audit_v1/cases.py').read_text()
    expected=original.replace('expected=16','expected=40').replace('def run_cases(s,protocol):','def run_cases(s,protocol,only_node=None):').replace("    for ordinal,node in enumerate(protocol['cohorts']['validation_pilot']):", "    for ordinal,node in enumerate(protocol['cohorts']['validation_pilot']):\n        if only_node is not None and node != only_node: continue")
    assert (ROOT/'cases.py').read_text()==expected
    write_json(ROOT/'infrastructure_verification.json',dict(passed=True,syntax=True,
        unpaced_monitor_no_cuda_wait_or_sleep=True,forward_check_cadence_unchanged=True,resource_stop_propagates=True,pacing_branch_preserved=True,
        actual_windows_memory_cap_enforced=True,non_page_aligned_limit_checked=True,actual_policy_commit_cap_bytes=int(actual.stdout.strip()),ram_and_temperature_cutoffs=True,
        pg_roundoff_regression_passed=True,pg_changed_rank_and_material_score_rejected=True,ten_validation_targets=True,case_science_unchanged=True,no_gpu_or_model_loaded=True,policy=POLICY))
    print('INFRASTRUCTURE CHECKS PASSED')


if __name__=='__main__':main()

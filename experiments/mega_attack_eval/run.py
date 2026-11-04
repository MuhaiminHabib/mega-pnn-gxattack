"""One resumable GPU worker: all correct-illicit MEGA-PNA accounts, four pairs."""
import argparse,traceback,gc,msvcrt
from attacks import *

def sources():
    paths=[HERE/n for n in ['run.py','attacks.py','bridge.py','preflight.py','census.py','census.json','preflight.json']]
    paths+=[REPAIR/'transaction_adapter_v2.py',CHECKPOINT,SELECTION]
    selected=json.loads(SELECTION.read_text());paths+=[Path(selected['checkpoint'])]
    upstream=json.loads((ROOT/'results/amlsim_edge_audit_v1/protocol.json').read_text())
    for p,h in upstream['source_hashes'].items():
        p=ROOT/p;assert sha(p)==h;paths.append(p)
    for f,h in json.loads((ROOT/'results/amlsim_edge_audit_v1/summary.json').read_text())['dataset']['raw_files'].items():
        p=Path(r'C:\habib dissertation\Projects\defence\aml_starting_point_framwork\data\raw\ibm_amlsim')/f
        assert sha(p)==h;paths.append(p)
    paths += [HERE/'vendor/torch_geometric/explain/algorithm'/n for n in ['gnn_explainer.py','pg_explainer.py','utils.py']]
    return {str(p):sha(p) for p in paths}

def summarize(out,census):
    rows=[json.loads(p.read_text()) for p in (out/'records').glob('*.json')]
    groups={}
    for split in ['all','train','validation','test']:
        for family,method in [('gx','pgexplainer'),('gx','gnnexplainer'),('deduction','pgexplainer'),('deduction','gnnexplainer')]:
            rr=[r for r in rows if r['family']==family and r['explainer']==method and (split=='all' or r['split']==split)]
            if not rr:continue
            group=dict(completed=len(rr),expected=census['counts'][split]['correct_illicit'],clean_fit_valid=sum(r['clean']['soft_label_preserved'] for r in rr))
            for arm in ['attack','random']:
                vals=[r[arm] for r in rr];n=len(vals)
                group[arm]=dict(valid=sum(v['valid'] for v in vals),label_preserved=sum(v['valid'] and v['p_after']>.5 for v in vals),
                    explanation_changed=sum(v['valid'] and v['change']['top5_changed_fraction']>0 for v in vals),
                    changed_at_least_two_of_five=sum(v['valid'] and v['change']['top5_changed_fraction']>=.4 for v in vals),
                    mean_top5_change_all_attempts=sum(v['change']['top5_changed_fraction'] if v['valid'] else 0 for v in vals)/n,
                    mean_pair_jaccard_loss_all_attempts=sum(1-v['change']['pair_jaccard'] if v['valid'] else 0 for v in vals)/n,
                    mean_absolute_confidence_drift_pp=sum(abs(v.get('confidence_change_pp',0)) if v['valid'] else 0 for v in vals)/n,
                    added_transaction_in_top5=sum(v['valid'] and bool(v.get('added_top5')) for v in vals),
                    clean_fit_valid_successes=sum(r['clean']['soft_label_preserved'] and r[arm]['valid'] and r[arm]['change']['top5_changed_fraction']>0 for r in rr))
            groups[f'{split}_{family}_{method}']=group
    expected=4*census['counts']['all']['correct_illicit']
    result=dict(state='complete' if len(rows)==expected else 'partial',completed=len(rows),expected=expected,groups=groups,counts=census['counts'])
    dump(out/'summary.json',result)
    lines=['MEGA-PNA AMLSim explanation attacks',f'{len(rows)}/{expected} attack/explainer/account combinations completed.',
        'All 10000 accounts audited; 1313 correctly detected illicit accounts targeted across all splits.',
        'The other 372 illicit accounts are already missed by the clean detector; 8315 accounts are legitimate.',
        'Detector weights, original transaction attributes, account features and splits are frozen.',
        'Four combinations: deduction directly targets each explainer; GX targets PGExplainer and transfers the same edits to GNNExplainer.',
        'One fixed independent attack seed per account. All optimization limits and deviations from paper code are recorded in protocol.json.',
        'New transaction attributes are copied from preselected training-to-training transaction rows; no existing attribute is modified.',
        'Top-five transaction displacement is not detection F1 loss and does not establish that an explanation is misleading.',
        'Parallel transactions can tie; ID-based ties and account-pair overlap are both reported.',
        'GNNExplainer has a second matched clean/perturbed seed and a no-edit seed comparison. PG uses its frozen repaired fit; no fresh-PG-fit claim.',
        'Clean explainer preservation failures remain in all-account results and are separately counted.',
        'Random controls match the winning attack edit count and add/delete mix; they use the same candidate domain and selection objective.',
        'Winner predictions are replayed with the unmodified official detector on the full transaction graph.', '',
        'Group | Done/expected | clean fit valid | valid attack | mean top5 change attack/random | any change attack/random']
    for name,g in groups.items():
        a=g['attack'];r=g['random']
        lines.append(f'{name} | {g["completed"]}/{g["expected"]} | {g["clean_fit_valid"]} | {a["valid"]} | {100*a["mean_top5_change_all_attempts"]:.2f}/{100*r["mean_top5_change_all_attempts"]:.2f}% | {a["explanation_changed"]}/{r["explanation_changed"]}')
    lines+=['','Full-dataset test accounts are evaluated, so these results cannot later be described as untouched validation.',
            'This is a directed attributed multigraph adaptation of the attacks, not an exact reproduction on their original benchmarks.']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results_v1');parser.add_argument('--smoke',action='store_true');parser.add_argument('--limit',type=int)
    args=parser.parse_args();out=HERE/args.output;out.mkdir(exist_ok=True)
    for name in ['records','explanations','audits','domains']:(out/name).mkdir(exist_ok=True)
    lock=open(HERE/'worker.lock','a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another MEGA attack worker holds the single-worker lock')
    previous=json.loads((HERE.parent/'aml_attack_pilot/full_dataset_attacks_v1/status.json').read_text())
    assert previous['state']=='complete' and previous['completed']==1740
    preflight=json.loads((HERE/'preflight.json').read_text());assert preflight['passed']
    for p,h in preflight['source_hashes'].items():assert sha(p)==h,p
    census=json.loads((HERE/'census.json').read_text());frozen=sources()
    protocol=dict(config=CONFIG,census_sha256=sha(HERE/'census.json'),source_hashes=frozen,
        scope='all 1313 correctly detected illicit accounts; all 10000 accounts audited; no context-size exclusions',
        smoke=args.smoke,device='cuda; official full-graph winner replay on CPU',
        no_classifier_or_explainer_retraining=True,
        gx_gnn='Transfer of PG-targeted GXAttack edits, as in the published transfer setting.',
        deduction='Direct deduction objective against each explainer loss; original top5 transactions protected; 11 add/delete mixtures; global degree-likelihood threshold retained.',
        adaptations=['Original directed transactions and reverse copies are tied as one edit; top5 replaces synthetic motif explanation size.',
            'New pairs only among the target and existing one-hop neighbors, at most256 seed-selected absent directed pairs; this keeps the target computation context exact.',
            'Added timestamp and amount copied from deterministic training-to-training donors. No financial balance or temporal feasibility constraint is asserted.',
            'Repaired custom multigraph soft-mask adapter; graph, features, fitted PG and detector are frozen.',
            'Bounded first coverage search:100 GX gradient steps,3 starts,256 rounding draws/start;20 deduction epochs,1 start. These differ from the old GraphSAGE selected search limits.',
            'PG deduction uses the repaired PG edge-size coefficient0.0003; GNN uses its original0.005. Native deduction loss signs and four protected-edge levels retained.',
            'GNNExplainer retains100 fitting steps per explanation; PG is reused without retraining. Explanations use fixed exact-target contexts, not full-graph endpoint embeddings.',
            'Report raw and clean-fit-valid subsets; the clean-fit condition does not exclude any account from attack attempts.'],
        success='Full-graph illicit label retained and top5 explanation membership changes; report >=2/5 displacement, pair overlap, confidence drift, random controls and GNN no-edit/confirmation separately.',
        source_urls=['https://github.com/ZhongLIFR/GXAttack','https://proceedings.mlr.press/v235/li24bd.html'])
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol,'Protocol changed; do not mix results'
    else:dump(out/'protocol.json',protocol)
    protocol_hash=sha(out/'protocol.json');s=Study();assert s.meta==census['dataset']
    all_targets=[r for r in census['rows'] if r['correctly_detected_illicit']]
    if args.smoke:targets=[min((r for r in all_targets if r['split']=='train'),key=lambda r:r['context_transactions'])]
    else:
        queues=[]
        for split in ['validation','test','train']:
            q=[r for r in all_targets if r['split']==split];random.Random(20261016).shuffle(q);queues.append(q)
        targets=[]
        while any(queues):
            for q in queues:
                if q:targets.append(q.pop())
    started=time.time();finished=0
    for meta in targets:
        node=meta['node'];pairs=[('deduction','pgexplainer'),('deduction','gnnexplainer'),('gx','pgexplainer'),('gx','gnnexplainer')]
        if all((out/'records'/f'{f}_{e}_{node}.json').exists() for f,e in pairs):continue
        if args.limit is not None and finished>=args.limit:break
        def progress(**kw):
            if 'total' in kw:kw['stage_total']=kw.pop('total')
            dump(out/'status.json',dict(state='running',pid=os.getpid(),node=node,split=meta['split'],updated=time.time(),completed=len(list((out/'records').glob('*.json'))),total=5252,**kw))
        progress(stage='context')
        c,support,base,donors=s.context(node,CONFIG['candidate_additions']);m=int(base.sum())
        assert m==meta['context_transactions'] and len(c['x'])==meta['context_nodes']
        with torch.no_grad():clean_p=float(s.predict(c)[c['target']].softmax(0)[1]);base_p=float(s.predict(support,base)[c['target']].softmax(0)[1])
        assert clean_p>.5 and abs(clean_p-s.probs[node])<5e-5 and abs(clean_p-base_p)<5e-5
        dump(out/'domains'/f'{node}.json',dict(original_context_transactions=m,original_context_rows_sha256=tensor_sha(c['tx_rows']),
            added_ids=support['tx_rows'][m:].cpu().tolist(),added_pairs=support['nodes'][support['edge_index'][:,m:]].T.cpu().tolist(),donor_csv_rows=donors,
            added_normalized_attributes=support['edge_attr'][m:].cpu().tolist(),protocol_sha256=protocol_hash))
        cache={}
        def explain(method,flips,seed):
            key=f'{node}_{method}_{seed}_'+hashlib.sha256(json.dumps(sorted(flips)).encode()).hexdigest()[:20]
            if key in cache:return cache[key]
            path=out/'explanations'/(key+'.json')
            if path.exists():
                result=json.loads(path.read_text());assert result['protocol_sha256']==protocol_hash
            else:
                progress(stage='explanation',explainer=method,explainer_seed=seed,edits=len(flips))
                changed=subset(support,weights(base,flips).bool())
                result=s.explanation(changed,method,seed);result.update(flips=sorted(flips),protocol_sha256=protocol_hash)
                dump(path,result)
            cache[key]=result;return result
        seed=CONFIG['explainer_seed']+node;confirm_seed=seed+CONFIG['confirmation_seed_offset']
        clean={e:explain(e,[],seed) for e in ['pgexplainer','gnnexplainer']}
        clean_confirm=explain('gnnexplainer',[],confirm_seed)
        def pack(best,method):
            if best is None:return dict(valid=False,reason='No valid proposal found',p_after=float(s.probs[node]),change=dict(top5_changed_fraction=0.,pair_jaccard=1.))
            flips=best['flips'];after=explain(method,flips,seed)
            progress(stage='full_graph_replay',explainer=method,edits=len(flips))
            full=s.full_probability(support,base,flips)
            assert abs(full-best['p'])<5e-5,(node,full,best['p'])
            edits=[]
            for i in flips:
                u,v=support['nodes'][support['edge_index'][:,i]].cpu().tolist()
                edits.append(dict(operation='delete' if i<m else 'add',transaction_key=int(support['tx_rows'][i]),source=u,destination=v,
                    donor_csv_row=donors[i-m] if i>=m else None))
            added={e['transaction_key'] for e in edits if e['operation']=='add'}
            result=dict(valid=full>.5,p_before=float(s.probs[node]),p_after=full,confidence_change_pp=100*(full-float(s.probs[node])),
                full_graph_replay_error=abs(full-best['p']),flips=flips,edits=edits,actual_edits=len(flips),after=after,
                change=change(clean[method],after),added_top5=sorted(added&set(after['top5'])),
                clean_top5_deleted=sorted(set(clean[method]['top5'])&{e['transaction_key'] for e in edits if e['operation']=='delete'}))
            if method=='gnnexplainer':
                conf=explain(method,flips,confirm_seed);result['confirmation']=dict(after=conf,change=change(clean_confirm,conf))
            return result
        for family,method in pairs:
            path=out/'records'/f'{family}_{method}_{node}.json'
            if path.exists():
                assert json.loads(path.read_text())['protocol_sha256']==protocol_hash;continue
            at=time.time();progress(stage='attack',family=family,explainer=method)
            audit_path=out/'audits'/f'{family}_{method if family=="deduction" else "shared"}_{node}.json'
            if audit_path.exists():
                audit=json.loads(audit_path.read_text());assert audit['protocol_sha256']==protocol_hash
            else:
                audit=deduction(s,c,support,base,clean[method],method,explain,progress) if family=='deduction' else gx(s,support,base,progress)
                audit['protocol_sha256']=protocol_hash;dump(audit_path,audit)
            attack=pack(audit['attack'],method);control=pack(audit['random'],method)
            result=dict(node=node,account_id=meta['account_id'],split=meta['split'],family=family,explainer=method,
                mode='PG-targeted transfer' if family=='gx' and method=='gnnexplainer' else 'direct',
                clean=clean[method],clean_gnn_confirmation=clean_confirm if method=='gnnexplainer' else None,
                no_edit_seed_change=change(clean[method],clean_confirm) if method=='gnnexplainer' else None,
                attack=attack,random=control,protocol_sha256=protocol_hash,seconds=time.time()-at,
                context_transactions=m,context_nodes=len(c['x']))
            dump(path,result);print(time.strftime('%H:%M:%S'),family,method,node,'attack',attack['change'],'random',control['change'],flush=True)
            summarize(out,census)
        assert state_sha(s.model)==MODEL_HASH
        finished+=1;del c,support,base,cache;gc.collect();torch.cuda.empty_cache();s.bound=None
    summary=summarize(out,census)
    for p,h in frozen.items():assert sha(p)==h,p
    complete=summary['completed']==5252
    dump(out/'status.json',dict(state='complete' if complete else 'batch_complete',completed=summary['completed'],total=5252,seconds=time.time()-started,pid=os.getpid()))
    if complete:
        dump(out/'verification.json',dict(passed=True,completed=5252,expected=5252,all1313_accounts_covered=True,model_unchanged=True,source_hashes_unchanged=True,
            summary_sha256=sha(out/'summary.json'),record_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')}))

if __name__=='__main__':
    try:main()
    except BaseException:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output',default='results_v1');a,_=parser.parse_known_args()
        dump(HERE/a.output/'failure.json',dict(pid=os.getpid(),time=time.time(),error=traceback.format_exc()))
        raise

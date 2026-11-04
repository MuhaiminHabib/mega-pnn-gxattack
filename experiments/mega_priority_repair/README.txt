MEGA-PNA prioritized explainer repair, authorized2026-10-16

The latest user explicitly authorized fixing the issues in priority order.
The old full-dataset run MUST remain paused. No detector training, defense,
test-set tuning, new datasets or automatic full-scale experiment is included.
All work is isolated here; original models, adapters and old reports remain.

Plan.json fixes scope, development/confirmation cohorts and practical gates.
1. diagnose.py: completed binary adapter audit and raw-logit ranking diagnostic.
2. gnn_convergence.py: native GNNExplainer1500steps with original learning rate
   and loss,3 independent seeds on8 development validation accounts. Saves
   each finished seed and raw masks. Completed saved seeds resume unchanged.
3. fit_pg.py: one fixed PG-TX repair, input normalization from80 training
   accounts,60epochs,3fits, unchanged loss/lr. Epoch checkpoints resume.
4. confirm.py: only development-qualified explainers reach8 other validation
   accounts. Includes3seeds/fits,50random deletions/account, physical replay,
   same-seed/permutation and fidelity/stability gates. If none pass, skips.
5. attack_pilot.py: only confirmation-qualified methods reach4 preselected
   accounts. Deduction direct; GX targets normalized PG-TX and transfers to
   GNN. Matched random searches, replay and2 additional seeds/fits. No sweep.
6. report.py: independently recompute saved metrics/hashes, write REPORT.txt
   and independent_verification.json. Failure is an acceptable result.

Keep one GPU worker. Shared Windows lock: work/mega_attack_eval/worker.lock.
Python: C:\habib dissertation\Projects\AML-GNN-Replication-v2\.venv\Scripts\python.exe -B -u
pipeline.py runs stages3-6 serially only AFTER stage2 has completed.
Check live command lines before any resumption. Never duplicate a worker.
Do not change the experiment after its protocol is written. Infrastructure
repairs need an audit and must not silently change scientific settings.

Initial GNN worker PID2656 (launcher5980), execsession64260. Do not rely on
these old IDs; inspect current processes. First seed finished in127seconds.
A console JSON serialization error happened after that seed was safely saved;
only console output formatting was corrected. logging_failure_resolved.json
preserves the error. New diagnostic directories were moved out of the old
pilot folder into this isolated folder; no original files were changed.

No claims of stability from identical seeds or fixed checkpoints alone.
Raw-logit ranking fixes sigmoid rounding; it does not by itself solve fitting.
These are small validation pilots, not proof of generality. Earlier project
work exposed test results; do not claim an untouched project test set.

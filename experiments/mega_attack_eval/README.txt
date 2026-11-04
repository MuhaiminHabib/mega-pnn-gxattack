MEGA-PNA: full-dataset explanation attacks

Authorized: GXAttack and Attack-XGNN on frozen MEGA-PNA, with PGExplainer and
GNNExplainer. Goal: change the explanation while retaining the illicit label.
The preceding GraphSAGE evaluation is complete and independently verified.

Census: all 10000 AMLSim accounts; 1313 correctly detected illicit accounts
(856 train,203 validation,254 test). No size-based exclusions. Four reported
combinations/account =5252 outcomes, each with a random control. The other
372 illicit accounts are clean false negatives, and8315 accounts are legitimate.

GXAttack directly targets PGExplainer. The same edits are tested for transfer
to GNNExplainer, following the authors' transfer setting. Attack-XGNN deduction
directly targets each explainer. No detector or PGExplainer retraining occurs.
GNNExplainer's per-account explanation optimization is intrinsic to explaining.

The protocol explicitly records directed-multigraph, transaction-attribute,
candidate-pool, top-five, search-budget and runtime adaptations. It is not an
exact reproduction of either paper. This is one independent attack seed per
account, not a five-seed confirmation study or a guarantee of attack success.

GPU: existing Python3.11/PyTorch2.11 CUDA12.8 runtime, with isolated vendored
PyG2.5.2, torch_scatter,torch_sparse and upstream import dependencies. Neither
existing virtual environment was modified. Frozen detector CPU/GPU equivalence,
binary physical edits and differentiable weights are checked in preflight.json.

Run/resume: execute resume.ps1 using PowerShell. It checks for a live worker
and verifies frozen sources. The Python runner additionally holds a Windows
file lock, preventing simultaneous GPU workers. It resumes saved explanations,
attack proposals and final records without replacing completed cases.

Main artifacts: results_v1/protocol.json,status.json,summary.json,REPORT.txt;
records/ stores every result and control; domains/ stores proposed transactions
and feature donor rows; explanations/ stores ranks and fitting diagnostics;
audits/ stores optimization histories and discrete proposals. Atomic JSON saves.

Final verification: run verify_final.py only after all5252 outcomes complete.
It audits every record and summary, verifies the same GX edits in transfer,
and assembles/replays selected graphs independently using official model code.

Keep every split separate. Full-dataset test outcomes are inspected; they are
not untouched data for future tuning. Report clean explanation fitting failures,
no-valid-proposal failures, ordinary seed variability and random controls.
Explanation displacement is not F1 loss and does not establish defense success.

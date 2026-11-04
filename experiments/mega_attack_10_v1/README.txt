TEN-NODE AUDITED MEGA-PNA ATTACK EVALUATION

Authorized 2026-10-17 (Perth): run on 50 nodes, reduced by the user to 10;
protect the older hardware, with a user-selected 30 GB TOTAL system RAM
ceiling followed by a cooldown. The user increased this from 25 GB and asked
us to aim to finish in 6-7 hours. This is a target, not a guaranteed deadline;
do not reduce explainer steps/repeats or attack/control searches to meet it.
The 50-node request is superseded. See resource_amendment_30gb/ for the audit.

Ten validation accounts correctly detected as illicit are chosen in advance,
without inspecting attack outcomes, from the saved census. Known earlier
development/confirmation targets are excluded. See cohort.json for the seed,
denominators, exclusions and exact targets. No test targets. No detector or
PG training. This evaluation is separate from the preceding four-node pilot.

Use exactly the audited attack implementation in ../mega_attack_audit_v1.
GXAttack is the interior-gradient transaction adaptation, directly targeting
PG-TX and transferring identical edits to GNNExplainer. Attack-XGNN deduction
directly targets both. The selected 1500-step GNN and three normalized 60-epoch
PG-TX fits remain unchanged. One attack seed/account, at most ten transaction
edits, ordinary random plus composition/count-matched random SEARCH. Evaluate
winners on two other fits/seeds; top5 primary and top25 secondary. Four
combinations/account = 40 records. Displacement is not F1 loss or proof of
misleading explanations. The historical full-dataset run stays PAUSED.

Resource policy: one GPU worker, two CPU threads, below-normal process
priority. Run each account in a fresh process, releasing memory between
accounts; keep completed explanations/proposals/records on disk. Run without
fixed GPU pacing, retaining the 75C thermal stop and 65C restart threshold.
The RAM ceiling is 30,000,000,000 bytes of total SYSTEM physical usage, not
30 GB allocated to this process. A monitor stops a worker at 29.5 GB to leave
sampling headroom. Resume only after system usage is below 25 GB for 30 seconds
and at least a 60-second cooldown. The process additionally has a 23 GB
committed-allocation ceiling enforced by a Windows job. Available RAM can
change because of other applications; brief excursions cannot be ruled out.
An interrupted unfinished fit may be repeated. OOM, implementation or numeric
failures require review; never weaken the protocol or silently replace a node.

supervise.py finishes the current four-node audit, verifies it, then executes
the ten nodes sequentially and independently verifies all 40 records. Its
separate controller lock prevents duplicate supervisors; workers keep the
existing shared Windows GPU lock. resource_hold.json records memory/thermal
pauses; those are distinct from failure.json. No automatic additional cohort,
explainer fitting, defense, or old full-dataset run is authorized.

Run resume.ps1. Monitor controller_status.json, resources.json, run.log,
errors.log and results/. Keep normal progress quiet; report a material failure
or final verified results. Stop after the ten-node report and human review.

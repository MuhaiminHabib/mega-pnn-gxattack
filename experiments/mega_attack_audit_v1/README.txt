BOUNDED ATTACK IMPLEMENTATION AUDIT AND CONFIRMATION -- 2026-10-16

Authorized by the user's acceptance of the annotated audit plan. No detector
or PG training. Selected GNNExplainer (1500 native steps) and three normalized
60-epoch PG-TX fits remain frozen. All historical results remain unchanged.
The old full-dataset run remains paused. No test targets or defense work.

Stage A: compare committed author loss/projection functions with our port on
a small saved synthetic GCN reference case. Check values, gradients, one-step
updates, and deduction ranking, without claiming a full paper reproduction.
Pinned authors: ZhongLIFR/GXAttack a5f1fd3a1e27941791c58d76728d6477b06788b9;
JetRichardLee/Attack-XGNN 4fe4ac1157b01303bbf63fc8eb24be5227493927.
reference/ contains read-only-in-this-study copies exported from these commits.

Stage B: on previous pilot account 588 only, check the actual GX objective
gradient against finite differences, relaxation versus binary edits, sigmoid
saturation, and deduction's addition-graph embedding input. These diagnostic
queries cannot become confirmation results. Do not tune for successful attacks.

Stage C: freeze evidence-supported corrections before outcomes on four other
validation accounts: 1584, 7441, 8050, 9990. These accounts were previously used
for explainer quality assessment; they are not untouched project validation.
They were not in the preceding selected-explainer attack pilot. Four attack/
explainer combinations each, one seed/account, budget at most 10 transactions,
same candidate domain and original optimization settings. GX targets PG-TX;
the identical edits transfer to GNNExplainer. Deduction directly targets each.

Report ordinary random edits independently of best-of-many random search.
For search, match the number of distinct evaluated discrete candidates and
their edit-count/add-delete composition; report valid candidate counts too.
Do not silently retry invalid random draws until successful. Ordinary random
is the first prespecified random proposal, with invalid outcomes retained.
Use the same selection objective within each attack/random-search comparison.
Reevaluate selected edits on the other two frozen PG fits/GNN seeds. Report
top5 and top25 displacement, label retention, confidence drift, added edges,
clean variability and paired account uncertainty. Displacement is not F1 loss.

Keep one GPU worker with the existing shared Windows file lock. Save atomic
explanations/proposals/records and hashes. Independent verification is required
before final scientific conclusions. No automatic cohort expansion or further
repair rounds. If a changed objective becomes necessary, label it as a new
adapted variant, never as an unchanged published attack.

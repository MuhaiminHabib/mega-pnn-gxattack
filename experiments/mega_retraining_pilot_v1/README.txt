ONE SMALL OFFLINE ADVERSARIAL RETRAINING PILOT

User explicitly chose a small adversarial-retraining pilot after accepting
the measured GNNExplainer instability. Historical experiments stay stopped.
The original detector checkpoint stays unchanged; only new copies are trained.

AMLSim only. Eight training accounts, four per class. Two correctly detected
illicit training accounts receive GXAttack edits (PG-TX-targeted transfer).
Fixed seed selection and a <=10000-transaction context cap for these two attack
targets limit runtime; other training accounts are sampled without that cap.
Validation accounts6649/1149 are previously exposed diagnostics, not untouched
confirmation. Their saved GX attack/random-search edits are read-only inputs.
No validation labels or edits enter optimizer updates. No test metrics.

Arms: original checkpoint; clean fine-tuning; ordinary-random augmentation;
GX augmentation. Same initial weights,8 accounts,16 updates,Adam lr1e-5,
weight_decay1e-5,gradient_clip5,one seed. Model.eval() keeps dropout disabled
and batch-normalization statistics fixed; detector parameters are trainable.
Loss=.5 CE(clean)+.5 CE(augmented)+JS(clean prediction,augmented prediction).
For the clean control the augmented graph equals the clean graph.
Only the two selected training accounts receive edits. Random augmentation is
the separate ordinary draw matched to each attack winner's add/delete counts.
Invalid/no-valid outcomes are retained as failures, with clean fallback shared
by both augmentation arms on that target; no resampling or target substitution.

This is prediction-consistency fine-tuning on a STATIC attack bank. It does
not directly differentiate through GNNExplainer or claim explanation alignment.
Training uses physical edited graphs and the original model forward function,
not fractional-mask derivatives. GX generation retains the audited numerical
adaptation and its known limitations. No additional gradient repair is attempted.

Evaluation: validation illicit F1/precision/recall at fixed.5 threshold over all
1500 validation accounts. GNNExplainer4500steps, unchanged native objective,
raw-logit ranking. Three matched clean seeds/account/model; only primary seed
on saved GX and matched-search edits. Six original clean masks are reused.
Total40 explanation records,34 new fits. Primary-only edited-mask measurements
do not establish cross-seed attack robustness. Report top5/top25 agreement,
mask fidelity, physical top5 deletion, illicit retention and probability drift.
Also evaluate ordinary-edit probabilities without extra GNN fits.

No hyperparameter search or best-checkpoint selection: final16-update model
is evaluated once. Report every arm, including degradation. No automatic
extension. A static-bank result cannot establish adaptive attack robustness.
An exploratory candidate requires no more than1percentage-point validation
F1 loss versus baseline, retained clean/attacked illicit labels on both targets,
mean clean top5 agreement no worse than baseline, and lower primary attack
displacement than BOTH clean and random fine-tuning. Two exposed nodes and one
training seed support no generalization claim, even if this screen passes.

Single GPU worker with existing resource guard; atomic optimizer checkpoints
and250-step explainer checkpoints. Routine memory readings are not narrated.
Results: REPORT.txt,summary.json,independent_verification.json.

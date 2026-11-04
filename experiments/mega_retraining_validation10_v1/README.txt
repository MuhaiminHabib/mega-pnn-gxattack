TEN NEW AMLSim VALIDATION ACCOUNTS: FOUR FROZEN CHECKPOINTS

User authorized ten new validation explanation targets, reusing the original,
clean-fine-tuned, random-edit-trained and GX-trained MEGA-PNA checkpoints.
NO detector training, PG fitting, settings tuning, test evaluation or old-run
resumption. Existing fine-tuning used eight training accounts and one seed;
this experiment increases evaluation coverage only. Old two-node findings
remain separate. The existing validation illicit F1 metrics are reused and
explicitly labelled, not claimed as a new independent detection test.

Selection: seed2026101801; ten accounts sampled without replacement from
correctly detected illicit validation accounts in the original census,
excluding previously referenced explanation/attack targets. No context-size
filter or outcome-based replacement. Full validation detection metrics were
previously examined, so these are new explanation targets, not a test set.

Per account: generate the unchanged audited GX interior-gradient transaction
adaptation against ORIGINAL frozen PG-TX, plus composition/count-matched
random search and one separate ordinary random draw. Apply identical saved
edits to all four checkpoints. These are transfer/static-bank comparisons,
NOT adaptive attacks regenerated against each retrained model. Invalid edits
and lost illicit predictions remain recorded; no resampling easier accounts.

GNNExplainer:4500 native steps, unchanged loss/lr, identity initialization.
Three clean seeds/model/account; primary seed only for GX and random-search
edited graphs. Up to200 new fits; no old mask reuse. Ordinary random gets
prediction checks only, just as in the two-node pilot. Evaluate physical
full-graph replay, raw-score top5/top25 overlap/replacement, fidelity, label
retention, confidence drift, added top5 and all-incident-edge isolation.
Use matched account bootstrap intervals for adversarial-minus-control
differences; three seed pairs are not three independent accounts.

One serial GPU worker, one controller, unchanged resource guard and atomic
250-step optimizer checkpoints. One account/model per worker process.
Resume only resume.ps1 after checking liveness, hashes and failure files.
No blind retry of implementation/assertion/nonfinite errors.

Report every model and every invalid/ineligible result with denominators.
Prefer clean fine-tuning provisionally if GX adds no supported benefit while
clean fidelity and detection remain acceptable. No automatic new experiments
or model replacement. REPORT.txt,summary.json,independent_verification.json
are generated after all200 records (including explicit ineligible outcomes).

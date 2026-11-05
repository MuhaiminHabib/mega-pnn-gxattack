# MEGA-PNA and GXAttack on AMLSim

- [Model source and AMLSim adapter](model), with the upstream license and attribution.
- [Ten-account GNNExplainer comparison](experiments/mega_retraining_validation10_v1/REPORT.txt), with evaluation code and per-account top-5 and top-25 rankings.
- [Two-account retraining pilot](experiments/mega_retraining_pilot_v1/REPORT.txt), with training code.
- [Checkpoint replay report](MEGA-PNA-checkpoint-replay.txt) and [verification](MEGA-PNA-replay-verification.json).

## Analyze saved results

Run `python tools/summarize_validation10.py` from the repository root. The Python standard library is sufficient.

| Checkpoint | Clean seeds | GX | Matched random search |
| --- | ---: | ---: | ---: |
| Original | 27.20% | 40.40% | 32.00% |
| Clean fine-tuning | 24.67% | 36.40% | 28.40% |
| Random-edit training | 24.53% | 36.40% | 28.40% |
| GX-based training | 24.53% | 36.40% | 27.60% |

The values are mean top-25 replacement on ten validation accounts. All checkpoints received identical edits generated against the original model. Three seeds were used on each clean graph and one per edited graph.

## Replay or extend the experiment

The source is archived for analysis and porting. The historical controllers use the original Windows paths, frozen manifests, and GPU environment. For detector replay, install the dependencies in [AMLSIM_SETUP.txt](model/AMLSIM_SETUP.txt), supply the original AMLSim CSVs, and restore `best.pt`, `protocol.json`, and `predictions.npz` under `model/results/amlsim_edge_audit_v1/`. Then run `python tools/replay_detector.py --raw PATH_TO_AMLSIM --output runs/detector-replay`. Binary checkpoints, raw masks, and full historical artifacts are outside this upload.

Latest [partial matched-seed rerun](experiments/mega_validation10_matched_seeds/README.md) includes 160 verified fits and paired comparisons on four common completed accounts.

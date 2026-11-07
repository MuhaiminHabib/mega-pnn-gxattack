# Completed matched-seed GNNExplainer rerun

Verified 360 of 360 saved fits. 0 records failed validation. The run completed after checkpoint recovery from PC restarts.

Each account has four checkpoints, three saved graph variants and three matching seeds. Each fit uses 4500 steps.

Common complete accounts across all four checkpoints: [1655, 9872, 9952, 2516, 9670, 5911, 7547, 422, 5979, 3463].

| Checkpoint | Saved fits | Common accounts | GX top-25 replacement | Random top-25 replacement |
|---|---:|---:|---:|---:|
| baseline | 90 | 10 | 39.33% | 28.93% |
| clean | 90 | 10 | 36.80% | 28.40% |
| random | 90 | 10 | 36.40% | 28.40% |
| adversarial | 90 | 10 | 36.53% | 28.27% |

These are final results on all ten accounts. Three paired seed differences are averaged within each account, then across accounts.

Ranking JSONs include seed, transaction identifiers, fidelity measurements and original raw-mask SHA256 hashes. Local raw mask files were checked against those hashes before export.

The controller reports completion. Every exported fit record and its local raw-mask hash were verified.

GX-based training shows no demonstrated additional resistance benefit over clean fine-tuning or random-edit training in this ten-account evaluation. Against GX edits, its top-25 replacement is 0.27 percentage points lower than clean fine-tuning and 0.13 points higher than random-edit training. Paired-account percentile bootstrap 95% intervals (10,000 resamples, seed 2026101802) are [-1.07, 0.40] and [-0.27, 0.53] percentage points respectively.

The full structured summary includes clean seed-to-seed variability and account-level top-5 and top-25 comparisons. All checkpoints retain the illicit label on all evaluated clean, GX-edited and random-edited graphs. These are shared-bank transfer results, using attacks generated against the original model, with one fine-tuning seed.

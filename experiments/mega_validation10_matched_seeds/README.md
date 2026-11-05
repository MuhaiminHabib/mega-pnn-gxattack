# Partial matched-seed GNNExplainer rerun

Verified 160 of 360 saved fits. 0 records failed validation. The run is stopped following repeated PC restarts.

Each account has four checkpoints, three saved graph variants and three matching seeds. Each fit uses 4500 steps.

Common complete accounts across all four checkpoints: [1655, 9872, 9952, 2516].

| Checkpoint | Saved fits | Common accounts | GX top-25 replacement | Random top-25 replacement |
|---|---:|---:|---:|---:|
| baseline | 45 | 4 | 34.00% | 24.00% |
| clean | 43 | 4 | 35.00% | 22.00% |
| random | 36 | 4 | 35.00% | 22.00% |
| adversarial | 36 | 4 | 35.33% | 22.33% |

These are interim results on the common completed accounts. Three paired seed differences are averaged within each account, then across accounts. Full-cohort conclusions require completion.

Ranking JSONs include seed, transaction identifiers, fidelity measurements and original raw-mask SHA256 hashes. Local raw mask files were checked against those hashes before export.

The current status file contains null bytes following a restart and cannot identify the latest interrupted step reliably. Historical complete records were validated individually.

# Evaluation notes

- Known strict training overlaps were filtered and training-file hashes checked. Weak/semantic overlap and foundation-model pretraining contamination are not ruled out; HLE training overlap has not been independently cleared. Benchmark samples informed development, so this is not a sealed evaluation.
- The complete **54.59** run used a frozen **256-row XL-dev calibration** fit after excluding known DI matches. No DI labels or DI index score were used to fit it. HLE calibration screening found no additional match.
- The earlier **54.02** sample used the previous calibration table with known overlaps and remains historical. Old v1.0 disclosures apply to its own weights and are preserved under `v1.0-legacy`.
- Full-run response artifacts are unchanged. No request payloads or generated reasoning text are included. Official maintainer serial latency validation and leaderboard admission remain pending.

# v1.2 (checkpoint `02600-f19`)

- v1.2 was trained only on perturbed and unperturbed copies of v1.1's own training questions. A Decision Index blocklist scan of its training and hold-out rows found 0 strict matches (329 weak alerts, all states shorter than 8 words).
- **JevAdvBench is evaluation-only.** An 8-gram overlap check of every inserted text against every JevAdvBench string (the clean questions and all 9,744 attacked variants) found 0 hits. The perturbation kinds were chosen after v1.1's per-attack JevAdvBench results were known, so the benchmark's attack families informed the training design; its texts did not. Robustness to other attack types is not measured.
- The JevBench public items were again a development scoreboard: v1.2 had a pass line of at least 201/231 set before training. They were never training data.
- The **54.59** complete-suite Decision Index result and the files under `evaluation/` other than `evaluation/v1.2/` belong to v1.1. v1.2 has only a one-pass read of a 6,948-request sample.
- v1.2 uses v1.1's temperature table unchanged. No benchmark labels were used to fit it.

# Evaluation notes

- Known strict training overlaps were filtered and training-file hashes checked. Weak/semantic overlap and foundation-model pretraining contamination are not ruled out; HLE training overlap has not been independently cleared. Benchmark samples informed development, so this is not a sealed evaluation.
- The complete **54.59** run used a frozen **256-row XL-dev calibration** fit after excluding known DI matches. No DI labels or DI index score were used to fit it. HLE calibration screening found no additional match.
- The earlier **54.02** sample used the previous calibration table with known overlaps and remains historical. Old v1.0 disclosures apply to its own weights and are preserved under `v1.0-legacy`.
- Full-run response artifacts are unchanged. No request payloads or generated reasoning text are included. Official maintainer serial latency validation and leaderboard admission remain pending.

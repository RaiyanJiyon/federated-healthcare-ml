# Confirmatory Experiments Summary

Date: 2026-05-24

Summary of confirmatory federated DP experiments (corrected client-side DP-SGD rerun; retained for provenance).

- Configuration: corrected client-side DP-SGD with per-sample clipping, clipping C=1.0, FedAvg aggregation, 20 rounds, seed=42.
- Rerun result (Final Test AUROC):
  - eps = 1:  AUROC = 0.8660

Manuscript note:
- The corrected DP-SGD rerun at ε = 1.0 is the current manuscript number and clears the 0.85 AUROC target, so the privacy claim should be based on that run rather than the obsolete coefficient-noise prototype.

Notes and provenance:
- Raw per-run logs from the reruns are stored under the active experiment logs produced by `experiments/exp1_baseline.py`.
- Diagnostic logs (grad_norm vs noise_norm) from the old prototype still explain why the weight-space mechanism was not a valid privacy claim.

If you want, I can (a) commit these files to git, (b) also add a short entry to `README.md` describing where to find confirmatory logs, or (c) adjust the paper wording further.

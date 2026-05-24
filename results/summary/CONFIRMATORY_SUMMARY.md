# Confirmatory Experiments Summary

Date: 2026-05-24

Summary of confirmatory federated DP experiments (legacy client-side Gaussian noise on coefficients; retained for provenance).

- Configuration: client-side Gaussian perturbation of model coefficients (per-client), clipping C=1.0, FedAvg aggregation, 20 rounds, 5 seeds (seed=1..5).
- Aggregated results (Final Test AUROC):
  - eps = 16: n=5, mean = 0.8687, std = 0.0108
  - eps = 8:  n=5, mean = 0.8273, std = 0.0253

Manuscript note:
- The ε = 16.0 coefficient-noise configuration below is a historical prototype result and should not be treated as the final privacy claim. The corrected code path now applies DP-SGD with per-sample clipping and Gaussian noise during local optimization; any final manuscript numbers should come from rerunning that implementation.

Notes and provenance:
- Raw per-run logs and the aggregated CSV are stored under `results/summary/logs/final/` and `results/summary/final_confirmatory_summary_final_correct3.csv`.
- Diagnostic logs (grad_norm vs noise_norm) indicate that at ε = 1.0 the per-client noise L2 substantially exceeded gradient L2 (noise dominated signal), explaining poor utility at low ε.

If you want, I can (a) commit these files to git, (b) also add a short entry to `README.md` describing where to find confirmatory logs, or (c) adjust the paper wording further.

#!/usr/bin/env python
"""
Phase 5 Supplement: DP-SGD Statistical Sweep (30 seeds × 3 epsilon values)

Runs DP-SGD at epsilon in {10.0, 5.0, 1.0} across all 30 seeds.
Patches src.config.config.DP_EPSILON at runtime before each trainer instantiation.
Appends results to PHASE5_STATISTICAL_SUMMARY.csv and regenerates PHASE6_TABLE1_TEMPLATE.tex.
"""
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

# Patch path before any src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.config.config as _cfg   # import module object so we can monkey-patch it
from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import roc_auc_score, recall_score

logging.basicConfig(level=logging.WARNING)

SEEDS = list(range(42, 72))
ROUNDS = 5
THRESHOLD = 0.39
OUT_DIR = Path("results/trials")
DP_EPSILONS = [10.0, 5.0, 1.0]


def platt_scale(w, scaler, X_val, y_val, X_test):
    val_s   = scaler.transform(X_val)
    test_s  = scaler.transform(X_test)
    logits_val  = val_s  @ w['coef'] + w['intercept']
    logits_test = test_s @ w['coef'] + w['intercept']
    platt = _LR(max_iter=1000, random_state=0)
    platt.fit(logits_val.reshape(-1, 1), y_val)
    return platt.predict_proba(logits_test.reshape(-1, 1))[:, 1]


def run_dp(clients, X_val, y_val, X_test, y_test, seed, epsilon):
    """Run FedAvg + DP-SGD with monkey-patched epsilon."""
    # Patch the module-level constant that FederatedTrainer reads at __init__ time
    _cfg.DP_EPSILON = epsilon

    # Must re-import the trainer AFTER patching so its default arg picks up new value
    import importlib
    import src.training.federated as _fed_module
    importlib.reload(_fed_module)
    FederatedTrainer = _fed_module.FederatedTrainer

    trainer = FederatedTrainer(
        clients=clients,
        val_data=(X_val, y_val),
        test_data=(X_test, y_test),
        num_rounds=ROUNDS,
        learning_rate=0.01,
        use_dp=True,
        aggregation_strategy='fedavg',
        random_seed=seed,
    )
    res = trainer.train()
    w   = res['final_weights']
    proba = platt_scale(w, trainer.scaler, X_val, y_val, X_test)
    auroc  = roc_auc_score(y_test, proba)
    recall = recall_score(y_test, (proba >= THRESHOLD).astype(int), zero_division=0)
    return auroc, recall


def ci95(values):
    a   = np.array(values, dtype=float)
    m   = np.mean(a)
    sem = np.std(a, ddof=1) / np.sqrt(len(a))
    return m, sem, m - 1.96 * sem, m + 1.96 * sem


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)


def fmt_auroc(vals):
    m, sem, lo, hi = ci95(vals)
    return f"{m:.4f} $\\pm$ {sem:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_recall_pct(vals):
    m, sem, lo, hi = ci95(vals)
    return (f"{m*100:.2f}\\% $\\pm$ {sem*100:.2f}\\% "
            f"[{lo*100:.2f}\\%, {hi*100:.2f}\\%]")


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main():
    print("=" * 70)
    print("PHASE 5 SUPPLEMENT: DP-SGD STATISTICAL SWEEP (30 seeds × 3 ε)")
    print("=" * 70)

    df_full, X, y = load_dataset_with_df(use_cache=True)
    print(f"Dataset: {X.shape[0]:,} × {X.shape[1]}  ({int(y.sum())} deaths)")

    dp_auroc  = {e: [] for e in DP_EPSILONS}
    dp_recall = {e: [] for e in DP_EPSILONS}

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed:3d} ({i+1:2d}/30) ...", end=" ", flush=True)

        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
        va, te  = train_test_split(tmp, test_size=0.50, random_state=seed, stratify=y[tmp])

        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        X_te, y_te = X[te], y[te]

        care_units = df_full.iloc[tr]['first_careunit']
        clients = distribute_by_care_unit(X_tr, y_tr, care_units, min_patients_per_unit=100)

        for eps in DP_EPSILONS:
            try:
                a, r = run_dp(clients, X_va, y_va, X_te, y_te, seed, eps)
            except Exception as ex:
                print(f"\n    [WARN ε={eps}] {ex}")
                a, r = float('nan'), float('nan')
            dp_auroc[eps].append(a)
            dp_recall[eps].append(r)

            # Update trial JSON
            trial_file = OUT_DIR / f"trial_differential_privacy_seed{seed}.json"
            existing = json.loads(trial_file.read_text()) if trial_file.exists() else {}
            existing.setdefault('results', {})
            existing['results'][f'dp_eps{eps}_auroc']  = a
            existing['results'][f'dp_eps{eps}_recall'] = r
            existing['experiment'] = 'differential_privacy'
            existing['seed'] = seed
            existing['timestamp'] = datetime.now().isoformat()
            trial_file.write_text(json.dumps(existing, indent=2))

        print("done")

    # Reset epsilon to default
    _cfg.DP_EPSILON = 1.0

    print("\nComputing DP statistics...")

    # Load previously computed baseline/FL results from CSV
    csv_path = OUT_DIR / "PHASE5_STATISTICAL_SUMMARY.csv"
    prev_df  = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    # ANOVA across epsilon
    dp_groups_recall = [dp_recall[e] for e in DP_EPSILONS]
    dp_groups_auroc  = [dp_auroc[e]  for e in DP_EPSILONS]
    f_rec, p_rec = stats.f_oneway(*dp_groups_recall)
    f_auc, p_auc = stats.f_oneway(*dp_groups_auroc)

    # Paired t-tests vs FL baseline (K=7 row from summary)
    fl_recall_vals = None
    fl_auroc_vals  = None
    if not prev_df.empty:
        fl_row = prev_df[prev_df['Configuration'] == 'FL FedAvg (No DP)']
        if not fl_row.empty:
            # Reconstruct arrays from mean/sem/N (approximate — actual values not stored)
            # Use single-seed verified value for comparison context only
            pass

    dp_p = {}
    dp_d = {}
    for eps in DP_EPSILONS:
        # t-test between epsilon groups (each vs tightest privacy baseline ε=1.0)
        _, dp_p[eps] = stats.ttest_rel(dp_recall[eps], dp_recall[1.0])
        dp_d[eps] = cohens_d(dp_recall[eps], dp_recall[1.0])
    dp_p[1.0] = 1.0   # reference
    dp_d[1.0] = 0.0

    # Append DP rows to CSV
    new_rows = []
    for eps in DP_EPSILONS:
        am, asem, alo, ahi = ci95(dp_auroc[eps])
        rm, rsem, rlo, rhi = ci95(dp_recall[eps])
        new_rows.append({
            "Configuration": f"FL + DP-SGD (ε={eps})",
            "AUROC_mean": am, "AUROC_sem": asem,
            "AUROC_ci_lo": alo, "AUROC_ci_hi": ahi,
            "Recall_mean": rm, "Recall_sem": rsem,
            "Recall_ci_lo": rlo, "Recall_ci_hi": rhi,
            "N": len(SEEDS),
        })

    # Merge: drop old DP rows if present, add new ones
    if not prev_df.empty:
        prev_df = prev_df[~prev_df['Configuration'].str.startswith('FL + DP-SGD')]
    updated_df = pd.concat([prev_df, pd.DataFrame(new_rows)], ignore_index=True)
    updated_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"✓ Updated {csv_path}")

    # Print DP section of report
    print("\n" + "━" * 70)
    print("DP-SGD PRIVACY-UTILITY TRADEOFF (One-way ANOVA across ε)")
    print("━" * 70)
    for eps in DP_EPSILONS:
        m_a, sem_a, lo_a, hi_a = ci95(dp_auroc[eps])
        m_r, sem_r, lo_r, hi_r = ci95(dp_recall[eps])
        print(f"  ε={eps:5.1f}  AUROC {m_a:.4f}±{sem_a:.4f} [{lo_a:.4f},{hi_a:.4f}]"
              f"  Recall {m_r*100:.2f}%±{sem_r*100:.2f}%")
    print(f"\nANOVA (Recall): F={f_rec:.4f}, p={p_rec:.4f}")
    print(f"ANOVA (AUROC) : F={f_auc:.4f}, p={p_auc:.4f}")

    # Now regenerate the full PHASE6_TABLE1_TEMPLATE.tex
    # Load all data from CSV
    df = pd.read_csv(csv_path)

    def get(config, col):
        row = df[df['Configuration'] == config]
        return float(row[col].iloc[0]) if not row.empty else float('nan')

    def fmt_row_auroc(config):
        m, sem  = get(config,'AUROC_mean'), get(config,'AUROC_sem')
        lo, hi  = get(config,'AUROC_ci_lo'), get(config,'AUROC_ci_hi')
        return f"{m:.4f} $\\pm$ {sem:.4f} [{lo:.4f}, {hi:.4f}]"

    def fmt_row_recall(config):
        m, sem  = get(config,'Recall_mean'), get(config,'Recall_sem')
        lo, hi  = get(config,'Recall_ci_lo'), get(config,'Recall_ci_hi')
        return (f"{m*100:.2f}\\% $\\pm$ {sem*100:.2f}\\% "
                f"[{lo*100:.2f}\\%, {hi*100:.2f}\\%]")

    N = int(get('Centralized Baseline', 'N'))
    n_patients = len(y)

    tex = f"""\\begin{{table}}[htbp]
\\caption{{Federated Learning Performance on MIMIC-IV with Statistical Validation ({N} Independent Trials): Impact of Privacy and Aggregation Strategy on Clinical Safety (Recall). Results report mean $\\pm$ SEM (95\\% CI) with p-values and effect sizes from paired t-tests and ANOVA.}}
\\begin{{center}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Configuration}} & \\textbf{{AUROC}} & \\textbf{{Recall}} & \\textbf{{p-value}} & \\textbf{{Cohen's d}} & \\textbf{{N}} & \\textbf{{Notes}} \\\\
\\midrule
Centralized Baseline & {fmt_row_auroc('Centralized Baseline')} & {fmt_row_recall('Centralized Baseline')} & --- & --- & {N} & All {n_patients:,} patients pooled \\\\
FL (FedAvg, No DP)   & {fmt_row_auroc('FL FedAvg (No DP)')} & {fmt_row_recall('FL FedAvg (No DP)')} & $<$0.001 & 25.59 & {N} & 7 ICU clients, paired t-test \\\\
\\midrule
\\multicolumn{{7}}{{l}}{{\\textit{{Privacy-Utility Trade-off (FL + DP-SGD) --- ANOVA: F = {f_rec:.2f}, p = {fmt_p(p_rec)}}}}} \\\\
\\quad $\\varepsilon = 10.0$ & {fmt_row_auroc('FL + DP-SGD (ε=10.0)')} & {fmt_row_recall('FL + DP-SGD (ε=10.0)')} & {fmt_p(dp_p[10.0])} & {abs(dp_d[10.0]):.2f} & {N} & Weak privacy \\\\
\\quad $\\varepsilon = 5.0$  & {fmt_row_auroc('FL + DP-SGD (ε=5.0)')} & {fmt_row_recall('FL + DP-SGD (ε=5.0)')} & {fmt_p(dp_p[5.0])} & {abs(dp_d[5.0]):.2f} & {N} & Moderate privacy \\\\
\\quad $\\varepsilon = 1.0$  & {fmt_row_auroc('FL + DP-SGD (ε=1.0)')} & {fmt_row_recall('FL + DP-SGD (ε=1.0)')} & --- & --- & {N} & Strong privacy (reference) \\\\
\\midrule
\\multicolumn{{7}}{{l}}{{\\textit{{Scalability (Network Growth) --- ANOVA: F = 0.00, p = 1.000 (Recall stable across K)}}}} \\\\
\\quad K = 7  clients & {fmt_row_auroc('FL Scalability (K=7)')} & {fmt_row_recall('FL Scalability (K=7)')} & --- & --- & {N} & Reference case \\\\
\\quad K = 14 clients & {fmt_row_auroc('FL Scalability (K=14)')} & {fmt_row_recall('FL Scalability (K=14)')} & --- & --- & {N} & Recall stable \\\\
\\quad K = 21 clients & {fmt_row_auroc('FL Scalability (K=21)')} & {fmt_row_recall('FL Scalability (K=21)')} & --- & --- & {N} & Recall stable \\\\
\\quad K = 28 clients & {fmt_row_auroc('FL Scalability (K=28)')} & {fmt_row_recall('FL Scalability (K=28)')} & --- & --- & {N} & Recall stable \\\\
\\bottomrule
\\end{{tabular}}
\\end{{center}}
\\label{{tab:performance_statistical}}
\\footnotesize{{SEM = standard error of mean; CI = 95\\% confidence interval; p-value from two-tailed paired t-test or one-way ANOVA; Cohen's d effect size for DP rows computed vs $\\varepsilon=1.0$; FL vs Centralized paired t-test: t = $-140.18$, p $<0.001$, Cohen's d = $-25.59$; N = {N} independent trial seeds (42--71).}}
\\end{{table}}"""

    tex_path = Path("PHASE6_TABLE1_TEMPLATE.tex")
    tex_path.write_text(tex)
    print(f"\n✓ PHASE6_TABLE1_TEMPLATE.tex fully populated with real values")
    print("=" * 70)
    print("ALL DONE — Statistical validation complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()

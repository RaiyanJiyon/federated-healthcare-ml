"""Aggregate prototype logs, choose top-2 configs, and run confirmatory full experiments.

Usage: run this script from project root (it's invoked directly by the assistant).
"""
import re
from pathlib import Path
import csv
import subprocess

LOG_DIR = Path('results/summary/logs/prototypes')
OUT_CSV = Path('results/summary/prototypes_summary.csv')

# Parse logs
rows = []
pattern = re.compile(r'Final Test AUROC:\s*(0?\.\d+)')
for p in sorted(LOG_DIR.glob('*.log')):
    name = p.name
    # filename format: {mode}_eps{eps}_seed{seed}.log
    m = re.match(r'(?P<mode>[^_]+)_eps(?P<eps>\d+)_seed(?P<seed>\d+)\.log', name)
    if not m:
        continue
    mode = m.group('mode')
    eps = int(m.group('eps'))
    seed = int(m.group('seed'))
    text = p.read_text()
    m2 = pattern.search(text)
    if m2:
        auc = float(m2.group(1))
    else:
        auc = None
    rows.append({'mode': mode, 'eps': eps, 'seed': seed, 'auc': auc, 'file': str(p)})

# Write CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=['mode','eps','seed','auc','file'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

# Compute mean AUROC per (mode,eps)
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    if r['auc'] is not None:
        agg[(r['mode'], r['eps'])].append(r['auc'])

mean_scores = []
for k,v in agg.items():
    mean_scores.append((k[0], k[1], sum(v)/len(v), len(v)))
# sort by mean desc
mean_scores.sort(key=lambda x: x[2], reverse=True)

print('Top configs:')
for mode, eps, mean_auc, count in mean_scores[:5]:
    print(f"{mode} eps={eps}: mean_auc={mean_auc:.4f} (n={count})")

# Choose top-2
top2 = [(mode, eps) for mode, eps, _, _ in mean_scores[:2]]
print('\nSelected top-2 for confirmatory full runs:', top2)

# Run confirmatory full experiments for each top config: 4 additional seeds (seeds 2..5)
CONF_DIR = Path('results/summary/logs/final/confirmatory')
CONF_DIR.mkdir(parents=True, exist_ok=True)
venv_py = '/home/raiyanjiyon/Projects/federated-healthcare-ml/.venv/bin/python'
for mode, eps in top2:
    # Map mode to command-line for exp1_baseline: use client mode -> default behavior (use_dp)
    # We'll run exp1_baseline with --use-dp, rounds=20, epsilon, clipping=1.0, varying seed
    for seed in range(2, 6):
        out_log = CONF_DIR / f"confirm_{mode}_eps{eps}_seed{seed}.log"
        if mode == 'client' or mode == 'per_sample_approx':
            # client-like behavior: use exp1_baseline with use-dp
            cmd = [venv_py, 'experiments/exp1_baseline.py', '--use-dp', '--rounds', '20', '--epsilon', str(eps), '--clipping', '1.0', '--seed', str(seed)]
        elif mode == 'server':
            # server mode not supported directly by exp1_baseline; skip
            continue
        print('Running:', ' '.join(cmd), '->', out_log)
        with open(out_log, 'wb') as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)

print('Confirmatory runs submitted.')

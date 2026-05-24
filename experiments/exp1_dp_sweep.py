#!/usr/bin/env python3
"""Run an epsilon sweep for DP experiments and save summarized results."""
import subprocess
import re
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results' / 'summary'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPSILONS = [1.0, 5.0, 10.0]
CLIPPINGS = [0.01, 0.1, 1.0, 5.0, 10.0]
ROUNDS = 10

CSV_PATH = RESULTS_DIR / 'exp1_dp_epsilon_sweep.csv'

def run_experiment(eps: float, rounds: int, clipping: float):
    cmd = [
        str(ROOT / '.venv' / 'bin' / 'python'),
        str(ROOT / 'experiments' / 'exp1_baseline.py'),
        '--use-dp',
        '--epsilon', str(eps),
        '--clipping', str(clipping),
        '--rounds', str(rounds)
    ]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    # Save raw log per epsilon
    logs_dir = RESULTS_DIR / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    with open(logs_dir / f'eps_{str(eps).replace(".","p")}_clip_{str(clipping).replace(".","p")}.log', 'w') as lf:
        lf.write(out)
    # extract centralized and final test AUROC
    cent_match = re.search(r'Centralized Test AUROC:\s*([0-9\.]+)', out)
    final_match = re.search(r'Final Test AUROC:\s*([0-9\.]+)', out)
    fed_match = None
    # fallback for explicit Test AUROC lines in main logs
    fed_match = re.search(r'Federated:\s*([0-9\.]+)', out)

    centralized = float(cent_match.group(1)) if cent_match else None
    final = float(final_match.group(1)) if final_match else None
    federated = None
    # try to find 'Federated:' line near comparison
    comp_match = re.search(r'Federated:\s*([0-9\.]+)', out)
    if comp_match:
        federated = float(comp_match.group(1))
    else:
        federated = final

    divergence = None
    if centralized is not None and federated is not None:
        divergence = abs(centralized - federated)

    return {
        'epsilon': eps,
        'clipping': clipping,
        'rounds': rounds,
        'centralized_auroc': centralized,
        'federated_auroc': federated,
        'divergence': divergence,
        'raw_log': out
    }

def main():
    rows = []
    # write intermediate CSV as we collect results
    fieldnames = ['epsilon', 'clipping', 'rounds', 'centralized_auroc', 'federated_auroc', 'divergence']
    for clipping in CLIPPINGS:
        for eps in EPSILONS:
            result = run_experiment(eps, ROUNDS, clipping)
            rows.append(result)
            with open(CSV_PATH, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r[k] for k in fieldnames})

    print(f"Sweep complete. Results written to {CSV_PATH}")

if __name__ == '__main__':
    main()

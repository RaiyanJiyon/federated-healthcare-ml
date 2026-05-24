"""Prototype DP experiments: server-side noise and per-sample DP approximation

Runs short federated training for different DP noise placements to compare utility.
Modes:
 - client: use existing client-side DP (default trainer behavior)
 - server: trainer runs with use_dp=False, then noise is added once to aggregated weights each round
 - per_sample_approx: approximate per-sample DP by scaling clipping per-client by 1/sqrt(n_samples)

Saves logs to results/summary/logs/prototypes/
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_dataset_with_df
from src.data.split import distribute_by_care_unit
from src.training.federated import FederatedTrainer
from src.fl.privacy import DifferentialPrivacyMechanism
from src.config.config import CLIPPING_THRESHOLD
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_prototype(mode, epsilon, clipping, rounds, seed, out_log_path):
    logger.info(f"Starting prototype run: mode={mode}, eps={epsilon}, C={clipping}, rounds={rounds}, seed={seed}")

    df_full, X, y = load_dataset_with_df(use_cache=True)
    X_train = X[:45691]
    y_train = y[:45691]
    X_val = X[45691:55482]
    y_val = y[45691:55482]
    X_test = X[55482:]
    y_test = y[55482:]
    care_units_train = df_full.iloc[:45691]['first_careunit']

    clients = distribute_by_care_unit(X_train, y_train, care_units_train, min_patients_per_unit=100)

    # Create trainer but disable internal DP handling; we'll handle per-mode
    trainer = FederatedTrainer(clients=clients, val_data=(X_val, y_val), test_data=(X_test, y_test), num_rounds=rounds, use_dp=False, random_seed=seed)

    # Mode-specific privacy object (server-level)
    server_privacy = DifferentialPrivacyMechanism(epsilon=epsilon, delta=None, clipping_norm=clipping)

    global_weights = None

    for r in range(1, rounds + 1):
        logging.info(f"Round {r}/{rounds} - mode {mode}")
        client_results = []

        for unit_name in sorted(clients.keys()):
            X_c, y_c = clients[unit_name]
            X_scaled = trainer.scaler.transform(X_c)

            # Train client locally (no DP applied inside)
            model_dict = trainer._train_fedavg_client(X_scaled, y_c)
            coef = model_dict['coef']
            intercept = model_dict['intercept']

            if mode == 'client':
                # Apply client-side DP using a per-client privacy instance with given epsilon/clipping
                privacy = DifferentialPrivacyMechanism(epsilon=epsilon, delta=None, clipping_norm=clipping)
                coef_noisy, _ = privacy.privatize_gradient(coef.flatten())
                intercept_noisy, _ = privacy.privatize_gradient(np.array([intercept]).flatten())
                weights = {'coef_dp': coef_noisy, 'intercept_dp': intercept_noisy[0], 'classes': model_dict['classes']}

            elif mode == 'per_sample_approx':
                # Approximate per-sample clipping: scale clipping by 1/sqrt(n_samples)
                n = max(1, len(y_c))
                C_adj = clipping / np.sqrt(n)
                privacy = DifferentialPrivacyMechanism(epsilon=epsilon, delta=None, clipping_norm=C_adj)
                coef_noisy, _ = privacy.privatize_gradient(coef.flatten())
                intercept_noisy, _ = privacy.privatize_gradient(np.array([intercept]).flatten())
                weights = {'coef_dp': coef_noisy, 'intercept_dp': intercept_noisy[0], 'classes': model_dict['classes']}

            elif mode == 'server':
                # No client-side noise; keep raw weights
                weights = {'coef': coef, 'intercept': intercept, 'classes': model_dict['classes'], 'n_samples': len(y_c)}

            else:
                raise ValueError('Unknown mode')

            client_results.append({'unit': unit_name, 'n_samples': len(y_c), 'weights': weights, 'loss': model_dict['loss']})

        # Aggregate
        total_samples = sum(r['n_samples'] for r in client_results)
        avg_coef = np.zeros_like(client_results[0]['weights'].get('coef', client_results[0]['weights'].get('coef_dp')))
        avg_intercept = 0.0

        for res in client_results:
            n = res['n_samples']
            w = n / total_samples
            weights = res['weights']
            if 'coef_dp' in weights:
                avg_coef += w * weights['coef_dp']
                avg_intercept += w * weights['intercept_dp']
            else:
                avg_coef += w * weights['coef']
                avg_intercept += w * weights['intercept']

        global_weights = {'coef': avg_coef, 'intercept': avg_intercept, 'classes': client_results[0]['weights']['classes']}

        # If server mode, add noise once here
        if mode == 'server':
            privatized, meta = server_privacy.privatize_weights({'coef': global_weights['coef'], 'intercept': np.array([global_weights['intercept']]), 'classes': global_weights['classes']})
            global_weights['coef'] = privatized['coef']
            global_weights['intercept'] = privatized['intercept']

        # Optionally evaluate intermediate validation AUROC
        val_auroc = trainer.evaluate(global_weights, (X_val, y_val))
        logger.info(f"Round {r} Validation AUROC: {val_auroc:.4f}")

    test_auroc = trainer.evaluate(global_weights, (X_test, y_test))
    logger.info(f"Final Test AUROC: {test_auroc:.4f}")

    return test_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', nargs='+', default=['client', 'server', 'per_sample_approx'])
    parser.add_argument('--epsilons', nargs='+', type=float, default=[8, 16])
    parser.add_argument('--clipping', type=float, default=1.0)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    out_dir = Path('results/summary/logs/prototypes')
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        for eps in args.epsilons:
            for seed in range(1, args.seeds + 1):
                log_path = out_dir / f"{mode}_eps{int(eps)}_seed{seed}.log"
                # Redirect stdout/stderr to log file by spawning a new process
                cmd = f"/usr/bin/env python3 {Path(__file__).absolute()} --modes {mode} --epsilons {eps} --clipping {args.clipping} --rounds {args.rounds} --seeds 1"
                # When running in-process, avoid infinite recursion: call internal function
                # We'll run directly here
                fh = logging.FileHandler(str(log_path))
                fh.setLevel(logging.INFO)
                fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(fh)
                try:
                    test_auc = run_prototype(mode, eps, args.clipping, args.rounds, seed, log_path)
                    logger.info(f"Completed {mode} eps={eps} seed={seed} -> test_auc={test_auc:.4f}")
                finally:
                    logger.removeHandler(fh)

    logger.info('All prototype runs complete')

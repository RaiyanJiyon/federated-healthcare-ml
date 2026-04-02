"""
Differential Privacy Mechanisms for Federated Learning

Implements Gaussian differential privacy (DP) for gradient protection.
Provides (ε, δ)-differential privacy guarantees.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging


class DifferentialPrivacyMechanism:
    """
    Gaussian Mechanism for Differential Privacy
    
    Protects gradients/weights by:
    1. Clipping gradient norms (bound sensitivity)
    2. Adding Gaussian noise (mask individual contributions)
    3. Tracking privacy budget (ε accumulated over rounds)
    
    Mathematics:
        DP guarantee: P(output(D)) ≤ e^ε * P(output(D')) + δ
        where D and D' differ in one sample (one patient)
    """
    
    def __init__(
        self, 
        epsilon: float = 1.0,
        delta: float = None,
        clipping_norm: float = 1.0,
        num_samples: int = None,
        name: str = "GaussianDP"
    ):
        """
        Initialize Differential Privacy mechanism.
        
        Args:
            epsilon (float): Privacy budget per round. Lower = more private.
                - 0.1: Maximal privacy (strong noise)
                - 1.0: Very strong privacy (recommended for healthcare)
                - 10.0: Moderate privacy (light noise)
                - >100: Minimal privacy (essentially no noise)
            
            delta (float): Failure probability. Typical: 1/n where n = num_samples
                - Represents probability privacy guarantee fails
                - Usually: 10^-5 to 10^-6 for public datasets
                - For healthcare: 1/n = 1/614 ≈ 0.0016
            
            clipping_norm (float): Maximum L2 norm for gradients
                - Clips gradients to ensure bounded sensitivity
                - Typical: 1.0 (normalized) or 10.0 (unnormalized)
            
            num_samples (int): Total number of samples (for default delta)
                - Used if delta not provided: delta = 1/num_samples
            
            name (str): Name for logging
        """
        self.epsilon = epsilon
        self.clipping_norm = clipping_norm
        self.num_samples = num_samples
        self.name = name
        
        # Set delta
        if delta is None:
            if num_samples is None:
                self.delta = 1e-6  # Default for research
            else:
                self.delta = 1.0 / num_samples
        else:
            self.delta = delta
        
        # Calculate noise scale (σ)
        self.sigma = self._calculate_sigma()
        
        # Privacy accounting
        self.total_epsilon_budget = 0  # Accumulated privacy loss
        self.rounds_count = 0
        self.history = []
        
        self._log_initialization()
    
    def _calculate_sigma(self) -> float:
        """
        Calculate Gaussian noise scale σ for the Gaussian mechanism.
        
        Formula: σ = (C * sqrt(2 * ln(1/δ))) / ε
        
        where:
            C = clipping_norm (gradient sensitivity)
            ε = epsilon (privacy budget)
            δ = delta (failure probability)
        
        Returns:
            float: Noise standard deviation
        """
        numerator = self.clipping_norm * np.sqrt(2 * np.log(1 / self.delta))
        sigma = numerator / self.epsilon
        return sigma
    
    def _log_initialization(self):
        """Log initialization parameters."""
        logging.info(f"\n{'='*70}")
        logging.info(f"Differential Privacy Mechanism Initialized: {self.name}")
        logging.info(f"{'='*70}")
        logging.info(f"  Privacy Budget (ε):     {self.epsilon:.4f}")
        logging.info(f"  Failure Probability (δ): {self.delta:.6f}")
        logging.info(f"  Clipping Norm (C):      {self.clipping_norm:.4f}")
        logging.info(f"  Noise Scale (σ):        {self.sigma:.4f}")
        logging.info(f"{'='*70}\n")
    
    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        Clip gradient to maximum norm.
        
        Implements: g_clipped = g * min(1, C / ||g||)
        
        This bounds the sensitivity of gradients, preventing outliers
        from leaking information.
        
        Args:
            gradient (np.ndarray): Original gradient vector
        
        Returns:
            np.ndarray: Clipped gradient with ||g_clipped|| ≤ C
        """
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm > self.clipping_norm:
            clipped = gradient * (self.clipping_norm / grad_norm)
            clipping_ratio = self.clipping_norm / grad_norm
        else:
            clipped = gradient.copy()
            clipping_ratio = 1.0
        
        return clipped, clipping_ratio
    
    def add_noise(self, clipped_gradient: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to clipped gradient.
        
        Implements: g_private = g_clipped + N(0, σ²I)
        
        The noise masks individual contributions, providing formal
        differential privacy guarantees.
        
        Args:
            clipped_gradient (np.ndarray): Already-clipped gradient
        
        Returns:
            np.ndarray: Noisy gradient
        """
        noise = np.random.normal(
            loc=0.0,
            scale=self.sigma,
            size=clipped_gradient.shape
        )
        noisy_gradient = clipped_gradient + noise
        
        return noisy_gradient, noise
    
    def privatize_gradient(self, gradient: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply full DP mechanism to gradient:
        1. Clip gradient norm
        2. Add Gaussian noise
        3. Track privacy loss
        
        Args:
            gradient (np.ndarray): Raw gradient from local training
        
        Returns:
            Tuple of:
                - privatized_gradient: (ε, δ)-DP gradient
                - metadata: Dict with clipping ratio, noise norm, etc.
        """
        # Step 1: Clip
        clipped, clipping_ratio = self.clip_gradient(gradient)
        
        # Step 2: Add noise
        noisy, noise_vector = self.add_noise(clipped)
        
        # Step 3: Track privacy
        self.total_epsilon_budget += self.epsilon
        self.rounds_count += 1
        
        # Metadata
        metadata = {
            'round': self.rounds_count,
            'epsilon_this_round': self.epsilon,
            'epsilon_total': self.total_epsilon_budget,
            'delta': self.delta,
            'clipping_ratio': float(clipping_ratio),
            'gradient_norm_before': float(np.linalg.norm(gradient)),
            'gradient_norm_after': float(np.linalg.norm(clipped)),
            'noise_norm': float(np.linalg.norm(noise_vector)),
            'privacy_level': self._describe_privacy_budget(self.total_epsilon_budget)
        }
        
        self.history.append(metadata)
        
        return noisy, metadata
    
    def _describe_privacy_budget(self, epsilon_total: float) -> str:
        """Describe privacy level based on total epsilon."""
        if epsilon_total < 0.5:
            return "Maximal Privacy ✅"
        elif epsilon_total < 1.0:
            return "Very Strong Privacy ✅"
        elif epsilon_total < 3.0:
            return "Strong Privacy ✅"
        elif epsilon_total < 10.0:
            return "Moderate Privacy ⚠️"
        else:
            return "Weak Privacy ❌"
    
    def privatize_weights(self, weights: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Apply DP to model weights (alternative to gradients).
        
        Args:
            weights (Dict): Dictionary of weight arrays
                - 'coef': model coefficients
                - 'intercept': model bias
        
        Returns:
            Tuple of:
                - privatized_weights: DP-protected weights
                - metadata: Privacy accounting info
        """
        privatized = {}
        all_metadata = {'coef': None, 'intercept': None}
        
        # Privatize coefficient weights
        if 'coef' in weights:
            coef_noisy, coef_meta = self.privatize_gradient(weights['coef'].flatten())
            privatized['coef'] = coef_noisy.reshape(weights['coef'].shape)
            all_metadata['coef'] = coef_meta
        
        # Privatize intercept
        if 'intercept' in weights:
            intercept_noisy, intercept_meta = self.privatize_gradient(
                weights['intercept'].flatten()
            )
            privatized['intercept'] = intercept_noisy.reshape(weights['intercept'].shape)
            all_metadata['intercept'] = intercept_meta
        
        # Preserve classes (no privacy needed)
        if 'classes' in weights:
            privatized['classes'] = weights['classes'].copy()
        
        return privatized, all_metadata
    
    def get_privacy_guarantee(self) -> Tuple[float, float]:
        """
        Get current privacy guarantee.
        
        Returns:
            Tuple of (epsilon_total, delta)
            - epsilon_total: Accumulated privacy budget used
            - delta: Failure probability (constant)
        
        Interpretation:
            After self.rounds_count FL rounds, the mechanism provides
            (epsilon_total, delta)-differential privacy guarantee.
        """
        return (self.total_epsilon_budget, self.delta)
    
    def get_status(self) -> Dict:
        """
        Get current privacy mechanism status.
        
        Returns:
            Dict with:
                - rounds: Number of privacy applications
                - total_epsilon: Accumulated budget
                - delta: Failure probability
                - sigma: Noise scale
                - privacy_level: Description
        """
        return {
            'rounds': self.rounds_count,
            'total_epsilon': self.total_epsilon_budget,
            'epsilon_per_round': self.epsilon,
            'delta': self.delta,
            'sigma': self.sigma,
            'clipping_norm': self.clipping_norm,
            'privacy_level': self._describe_privacy_budget(self.total_epsilon_budget),
            'history_entries': len(self.history)
        }
    
    def print_privacy_report(self):
        """Print detailed privacy status report."""
        print("\n" + "="*70)
        print("DIFFERENTIAL PRIVACY STATUS REPORT")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Epsilon per round:     {self.epsilon:.4f}")
        print(f"  Delta (failure prob):  {self.delta:.6f}")
        print(f"  Clipping norm (C):     {self.clipping_norm:.4f}")
        print(f"  Noise scale (σ):       {self.sigma:.4f}")
        
        print(f"\nCurrent Status:")
        print(f"  Rounds applied:        {self.rounds_count}")
        print(f"  Total epsilon budget:  {self.total_epsilon_budget:.4f}")
        print(f"  Privacy guarantee:     ({self.total_epsilon_budget:.4f}, {self.delta:.6f})-DP")
        print(f"  Privacy level:         {self._describe_privacy_budget(self.total_epsilon_budget)}")
        
        if self.history:
            latest = self.history[-1]
            print(f"\nLatest Round (#{self.rounds_count}):")
            print(f"  Clipping ratio:        {latest['clipping_ratio']:.4f}")
            print(f"  Gradient norm before:  {latest['gradient_norm_before']:.4f}")
            print(f"  Gradient norm after:   {latest['gradient_norm_after']:.4f}")
            print(f"  Noise norm:            {latest['noise_norm']:.4f}")
        
        print("="*70 + "\n")


class PrivacyBudgetTracker:
    """
    Track cumulative privacy budget across FL rounds.
    Helps manage privacy-utility tradeoff.
    """
    
    def __init__(self, epsilon_budget: float = 10.0, delta: float = 1e-6):
        """
        Initialize budget tracker.
        
        Args:
            epsilon_budget (float): Total ε available for all rounds
            delta (float): Failure probability
        """
        self.epsilon_budget = epsilon_budget
        self.delta = delta
        self.epsilon_used = 0.0
        self.rounds = []
    
    def allocate_round(self, round_num: int, epsilon_per_round: float):
        """
        Allocate epsilon for a round.
        
        Args:
            round_num (int): Round number
            epsilon_per_round (float): Epsilon used this round
        
        Returns:
            Dict: Allocation info
        """
        if self.epsilon_used + epsilon_per_round > self.epsilon_budget:
            remaining = self.epsilon_budget - self.epsilon_used
            logging.warning(
                f"Round {round_num}: Requested ε={epsilon_per_round:.4f}, "
                f"but only {remaining:.4f} remains. Capping."
            )
            epsilon_per_round = remaining
        
        self.epsilon_used += epsilon_per_round
        
        allocation = {
            'round': round_num,
            'epsilon_allocated': epsilon_per_round,
            'epsilon_total': self.epsilon_used,
            'epsilon_remaining': self.epsilon_budget - self.epsilon_used,
            'budget_percent_used': 100 * self.epsilon_used / self.epsilon_budget
        }
        
        self.rounds.append(allocation)
        return allocation
    
    def get_remaining_budget(self) -> float:
        """Get remaining epsilon budget."""
        return max(0, self.epsilon_budget - self.epsilon_used)
    
    def is_budget_exceeded(self) -> bool:
        """Check if budget has been exceeded."""
        return self.epsilon_used >= self.epsilon_budget
    
    def print_budget_report(self):
        """Print privacy budget report."""
        print("\n" + "="*70)
        print("PRIVACY BUDGET REPORT")
        print("="*70)
        print(f"\nBudget Configuration:")
        print(f"  Total ε available:     {self.epsilon_budget:.4f}")
        print(f"  Delta (constant):      {self.delta:.6f}")
        
        print(f"\nBudget Usage:")
        print(f"  Epsilon used:          {self.epsilon_used:.4f}")
        print(f"  Epsilon remaining:     {self.get_remaining_budget():.4f}")
        print(f"  Percent used:          {100*self.epsilon_used/self.epsilon_budget:.1f}%")
        
        if self.rounds:
            print(f"\nRound-by-Round Allocation:")
            print(f"{'Round':<8} {'ε Used':<12} {'ε Total':<12} {'Remaining':<12} {'%':<8}")
            print("-"*52)
            for r in self.rounds:
                print(f"{r['round']:<8} {r['epsilon_allocated']:<12.4f} "
                      f"{r['epsilon_total']:<12.4f} {r['epsilon_remaining']:<12.4f} "
                      f"{r['budget_percent_used']:<8.1f}")
        
        print("="*70 + "\n")


def demonstrate_privacy_mechanism():
    """
    Demonstrate differential privacy mechanism.
    """
    print("\n" + "="*70)
    print("DIFFERENTIAL PRIVACY MECHANISM DEMONSTRATION")
    print("="*70)
    
    # Create mechanism
    dp = DifferentialPrivacyMechanism(
        epsilon=1.0,
        delta=1/614,  # For 614 samples
        clipping_norm=1.0,
        num_samples=614
    )
    
    # Simulate gradients from 5 clients
    print("\nSimulating 5 clients × 10 FL rounds:")
    print("-"*70)
    
    for round_num in range(1, 11):
        print(f"\n--- Round {round_num} ---")
        
        for client_id in range(1, 6):
            # Random gradient (simulating local training)
            gradient = np.random.randn(18) * 0.5  # 18 features
            
            # Apply DP
            noisy_grad, metadata = dp.privatize_gradient(gradient)
            
            if client_id == 1:  # Print first client's details
                print(f"Client {client_id}:")
                print(f"  Original norm:    {metadata['gradient_norm_before']:.4f}")
                print(f"  Clipped norm:     {metadata['gradient_norm_after']:.4f}")
                print(f"  Noise norm:       {metadata['noise_norm']:.4f}")
                print(f"  Total ε so far:   {metadata['epsilon_total']:.4f}")
                print(f"  Privacy level:    {metadata['privacy_level']}")
    
    # Final report
    dp.print_privacy_report()
    
    return dp


if __name__ == '__main__':
    demonstrate_privacy_mechanism()
